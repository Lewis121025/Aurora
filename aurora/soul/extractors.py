"""
aurora/soul/extractors.py
本模块定义了从文本中提取心理含义（Meaning）以及生成叙事文本（Narrative）的各种提供。
它实现了动态灵魂模型，将文本转换为多维度的心理证据。
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np

# 导入 LLM 提示词和模式
from aurora.integrations.llm.Prompt.generative_soul_prompt import (
    GEN_SOUL_AXIS_MERGE_SYSTEM_PROMPT,
    GEN_SOUL_DREAM_SYSTEM_PROMPT,
    GEN_SOUL_MEANING_SYSTEM_PROMPT,
    GEN_SOUL_MODE_LABEL_SYSTEM_PROMPT,
    GEN_SOUL_PERSONA_AXIS_SYSTEM_PROMPT,
    GEN_SOUL_REPAIR_SYSTEM_PROMPT,
    GEN_SOUL_SEMANTIC_PROJECTION_SYSTEM_PROMPT,
    GEN_SOUL_SUMMARY_SYSTEM_PROMPT,
    build_gen_soul_axis_merge_user_prompt,
    build_gen_soul_dream_user_prompt,
    build_gen_soul_meaning_user_prompt,
    build_gen_soul_mode_label_user_prompt,
    build_gen_soul_persona_axis_user_prompt,
    build_gen_soul_repair_user_prompt,
    build_gen_soul_semantic_projection_user_prompt,
    build_gen_soul_summary_user_prompt,
)
from aurora.integrations.llm.provider import LLMProvider
from aurora.integrations.llm.schemas import (
    AxisMergeJudgementPayload,
    DreamNarrationPayloadV4,
    MeaningFramePayloadV4,
    ModeLabelPayloadV4,
    NarrativeSummaryPayloadV4,
    PersonaAxisPayload,
    RepairNarrationPayloadV4,
    SemanticProjectionPayload,
)
from aurora.soul.models import (
    AxisSpec,
    EventFrame,
    IdentityState,
    MeaningExtraction,
    Message,
    NarrativeSummary,
    PsychologicalSchema,
    TextPart,
    axes_to_phrase,
    clamp,
    clamp01,
    heuristic_persona_axes,
    messages_text_only,
    messages_to_text,
    normalize_text,
    top_axes_description,
    tokenize_loose,
)


def _text_prompt_messages(*, system: str, user: str) -> tuple[Message, Message]:
    return (
        Message(role="system", parts=(TextPart(text=system),)),
        Message(role="user", parts=(TextPart(text=user),)),
    )


def _semantic_fallback(messages: Sequence[Message]) -> str:
    semantic_text = messages_text_only(messages)
    if semantic_text:
        return semantic_text
    return messages_to_text(messages, include_image_uris=True)


def _image_part_count(messages: Sequence[Message]) -> int:
    return sum(1 for message in messages for part in message.parts if getattr(part, "type", "") == "image")


def _salient_axis_names(state: IdentityState, *, limit: int = 4) -> List[str]:
    return [
        name
        for name, _ in sorted(state.axis_state.items(), key=lambda item: abs(item[1]), reverse=True)[
            :limit
        ]
    ]


def _bootstrap_mode_label(
    prototype_axes: Dict[str, float],
    schema: PsychologicalSchema,
    support: int,
) -> str:
    items = sorted(prototype_axes.items(), key=lambda item: abs(item[1]), reverse=True)
    words: List[str] = []
    for name, value in items[:2]:
        axis = schema.all_axes().get(name)
        if axis is None:
            continue
        words.append(axis.positive_pole if value >= 0 else axis.negative_pole)
    if not words:
        return f"mode_{support}"
    if len(words) == 1:
        return f"{words[0]} mode"
    return f"{words[0]} / {words[1]} mode"


class MeaningProvider(Protocol):
    """意义提取器接口协议，定义了从图文消息中提取 MeaningExtraction 的标准方法"""

    def project(self, messages: Sequence[Message]) -> str:
        """将结构化图文消息投影为规范化语义文本。"""
        ...

    def extract(
        self,
        messages: Sequence[Message],
        embedding: np.ndarray,
        schema: PsychologicalSchema,
        recent_tags: Optional[Sequence[str]] = None,
    ) -> MeaningExtraction:
        """从给定文本中提取心理语义框架"""
        ...

    def extract_persona_axes(self, profile_text: str) -> List[Dict[str, Any]]:
        """从一段描述性文本中自动发现并提取人设维度（Axis）"""
        ...

    def bootstrap_persona_axes(self, profile_text: str) -> List[Dict[str, Any]]:
        """启动阶段使用的快速 persona-axis 提取，不应依赖远程请求"""
        ...


class NarrativeProvider(Protocol):
    """叙事生成器接口协议，定义了生成自我总结、修复描述和梦境描述的标准方法"""

    def compose_summary(
        self,
        state: IdentityState,
        schema: PsychologicalSchema,
        recent_semantic_texts: Sequence[str],
    ) -> NarrativeSummary:
        """生成当前身份状态的总结性叙事"""
        ...

    def compose_repair(
        self,
        mode: str,
        state_before: IdentityState,
        state_after: IdentityState,
        dissonance_total: float,
        salient_axes: Sequence[str],
        plot_text: str,
        schema: PsychologicalSchema,
    ) -> str:
        """为身份修复（Identity Repair）过程生成叙事解释"""
        ...

    def compose_dream(
        self,
        operator: str,
        fragment_tags: Sequence[str],
        state: IdentityState,
        schema: PsychologicalSchema,
    ) -> str:
        """生成梦境（Dream）的描述文本"""
        ...

    def label_mode(
        self,
        prototype_axes: Dict[str, float],
        schema: PsychologicalSchema,
        support: int,
    ) -> str:
        """为涌现出的身份模式（Identity Mode）生成可理解的标签"""
        ...

    def bootstrap_mode_label(
        self,
        prototype_axes: Dict[str, float],
        schema: PsychologicalSchema,
        support: int,
    ) -> str:
        """启动阶段使用的快速 mode 命名，不应依赖远程请求"""
        ...

    def judge_axis_merge(
        self,
        canonical: AxisSpec,
        alias: AxisSpec,
        evidence_overlap: Sequence[str],
    ) -> tuple[bool, str]:
        """判断两个心理轴是否应当合并（用于 Schema 演化）"""
        ...


class HeuristicMeaningProvider:
    """基于规则和词库的启发式意义提取器。"""

    # 预定义的认知词库，用于快速判断基础倾向
    CARE_WORDS = {
        "care",
        "support",
        "listen",
        "comfort",
        "hug",
        "gentle",
        "warm",
        "陪",
        "安慰",
        "理解",
        "照顾",
        "拥抱",
        "温柔",
        "支持",
        "倾听",
    }
    THREAT_WORDS = {
        "attack",
        "betray",
        "lie",
        "control",
        "reject",
        "ignore",
        "hurt",
        "abandon",
        "manipulate",
        "攻击",
        "背叛",
        "欺骗",
        "控制",
        "拒绝",
        "不理",
        "伤害",
        "抛弃",
        "利用",
    }
    SHAME_WORDS = {
        "shame",
        "embarrass",
        "humiliate",
        "worthless",
        "失败",
        "羞耻",
        "丢脸",
        "没用",
        "糟糕",
    }
    AGENCY_WORDS = {
        "choose",
        "decide",
        "build",
        "refuse",
        "set boundary",
        "speak up",
        "行动",
        "选择",
        "决定",
        "建立",
        "拒绝",
        "边界",
        "表达",
    }
    CONTROL_WORDS = {
        "control",
        "force",
        "must",
        "can't",
        "manipulate",
        "管",
        "强迫",
        "必须",
        "不许",
        "操控",
    }
    ABANDON_WORDS = {
        "leave",
        "left",
        "distance",
        "ghost",
        "cold",
        "离开",
        "冷淡",
        "失联",
        "消失",
        "疏远",
    }
    POSITIVE_WORDS = {
        "safe",
        "trust",
        "good",
        "happy",
        "gentle",
        "kind",
        "可靠",
        "安心",
        "喜欢",
        "愉快",
        "幸福",
        "温暖",
    }
    NEGATIVE_WORDS = {
        "bad",
        "pain",
        "fear",
        "cry",
        "angry",
        "hurt",
        "awful",
        "害怕",
        "痛",
        "哭",
        "生气",
        "难过",
        "糟糕",
    }

    def extract_persona_axes(self, profile_text: str) -> List[Dict[str, Any]]:
        """基于简单规则的人设维度提取"""
        return heuristic_persona_axes(profile_text)

    def bootstrap_persona_axes(self, profile_text: str) -> List[Dict[str, Any]]:
        return self.extract_persona_axes(profile_text)

    def project(self, messages: Sequence[Message]) -> str:
        return _semantic_fallback(messages)

    def extract(
        self,
        messages: Sequence[Message],
        embedding: np.ndarray,
        schema: PsychologicalSchema,
        recent_tags: Optional[Sequence[str]] = None,
    ) -> MeaningExtraction:
        """
        启发式提取过程：
        1. 利用向量投影计算各轴得分。
        2. 基于词库匹配计算基础心理倾向（如 care, threat）。
        3. 综合计算效价（Valence）和唤醒度（Arousal）。
        """
        semantic_text = self.project(messages)
        text_lower = semantic_text.lower()
        tokens = set(tokenize_loose(semantic_text))

        # 核心：计算新输入在现有每一个心理轴上的投影得分
        axis_evidence = {
            name: axis.score(embedding, semantic_text) for name, axis in schema.all_axes().items()
        }

        # 计算各个基础维度的命中得分
        care = self._lex_score(text_lower, tokens, self.CARE_WORDS)
        threat = self._lex_score(text_lower, tokens, self.THREAT_WORDS)
        shame = self._lex_score(text_lower, tokens, self.SHAME_WORDS)
        agency_signal = self._lex_score(text_lower, tokens, self.AGENCY_WORDS)
        control = self._lex_score(text_lower, tokens, self.CONTROL_WORDS)
        abandonment = self._lex_score(text_lower, tokens, self.ABANDON_WORDS)
        pos = self._lex_score(text_lower, tokens, self.POSITIVE_WORDS)
        neg = self._lex_score(text_lower, tokens, self.NEGATIVE_WORDS)

        # 归一化计算
        valence = clamp(pos - neg)
        arousal = clamp01(0.25 + 0.45 * threat + 0.20 * shame + 0.15 * agency_signal + 0.15 * neg)
        self_relevance = 0.40
        if any(
            marker in text_lower
            for marker in ["我", "自己", "她", "你", "we", "i ", " me ", " my ", "you "]
        ):
            self_relevance += 0.18
        if "self" in text_lower:
            self_relevance += 0.12
        self_relevance = clamp01(self_relevance)

        # 自动生成语义标签
        tags = self._make_tags(
            text=semantic_text,
            axis_evidence=axis_evidence,
            care=care,
            threat=threat,
            shame=shame,
            agency_signal=agency_signal,
            control=control,
            abandonment=abandonment,
            recent_tags=recent_tags,
        )
        # 计算新颖度（基于轴激活数量）
        novelty = clamp01(
            0.12
            + 0.55
            * len([name for name, value in axis_evidence.items() if abs(value) > 0.25])
            / max(len(axis_evidence), 1)
            + 0.10 * max(0, len(set(tags)) - 3)
        )

        return MeaningExtraction(
            semantic_text=semantic_text,
            frame=EventFrame(
                axis_evidence=axis_evidence,
                valence=valence,
                arousal=arousal,
                care=care,
                threat=threat,
                control=control,
                abandonment=abandonment,
                agency_signal=agency_signal,
                shame=shame,
                novelty=novelty,
                self_relevance=self_relevance,
                tags=tuple(tags),
            ),
        )

    @staticmethod
    def _lex_score(text: str, tokens: set[str], lexicon: set[str]) -> float:
        """计算词库命中得分"""
        hits = 0
        for word in lexicon:
            if word in text or word.lower() in tokens:
                hits += 1
        if hits == 0:
            return 0.0
        return clamp01(0.22 * hits)

    def _make_tags(
        self,
        *,
        text: str,
        axis_evidence: Dict[str, float],
        care: float,
        threat: float,
        shame: float,
        agency_signal: float,
        control: float,
        abandonment: float,
        recent_tags: Optional[Sequence[str]],
    ) -> List[str]:
        """基于激活度和匹配词生成语义标签"""
        tags: List[str] = []
        # 将得分最高的前三个轴加入标签
        for name, value in sorted(
            axis_evidence.items(), key=lambda item: abs(item[1]), reverse=True
        )[:3]:
            if abs(value) > 0.18:
                tags.append(name)
                tags.append(f"{name}:{'pos' if value >= 0 else 'neg'}")
        if care > 0.2:
            tags.append("care")
        if threat > 0.2:
            tags.append("threat")
        if shame > 0.2:
            tags.append("shame")
        if agency_signal > 0.2:
            tags.append("agency")
        if control > 0.2:
            tags.append("control")
        if abandonment > 0.2:
            tags.append("abandonment")

        # 加入简单的分词结果作为内容标签
        for token in tokenize_loose(text)[:4]:
            if token not in tags:
                tags.append(token)
        # 回声标签（用于模拟注意力延续）
        for recent_tag in recent_tags or []:
            if recent_tag in tags:
                tags.append(f"echo:{recent_tag}")
        return tags[:12]


class LLMMeaningProvider:
    """基于大模型的意义提取器，能进行复杂的语义理解和定向轴评分"""

    def __init__(
        self,
        llm: LLMProvider,
        *,
        timeout_s: float = 12.0,
        max_retries: int = 1,
    ) -> None:
        self._llm = llm
        self._timeout_s = timeout_s
        self._max_retries = max_retries

    def extract_persona_axes(self, profile_text: str) -> List[Dict[str, Any]]:
        """利用 LLM 智能发现人设维度"""
        if not profile_text.strip():
            return []
        payload = self._llm.complete_json(
            messages=_text_prompt_messages(
                system=GEN_SOUL_PERSONA_AXIS_SYSTEM_PROMPT,
                user=build_gen_soul_persona_axis_user_prompt(profile_text=profile_text),
            ),
            schema=PersonaAxisPayload,
            temperature=0.1,
            timeout_s=self._timeout_s,
            metadata={"operation": "persona_axis_extraction"},
            max_retries=self._max_retries,
        )
        return [axis.model_dump() for axis in payload.axes]

    def bootstrap_persona_axes(self, profile_text: str) -> List[Dict[str, Any]]:
        return heuristic_persona_axes(profile_text)

    def project(self, messages: Sequence[Message]) -> str:
        semantic_text = _semantic_fallback(messages)
        image_part_count = _image_part_count(messages)
        if image_part_count == 0:
            return semantic_text
        payload = self._llm.complete_json(
            messages=(
                Message(
                    role="system",
                    parts=(TextPart(text=GEN_SOUL_SEMANTIC_PROJECTION_SYSTEM_PROMPT),),
                ),
                *messages,
                Message(
                    role="user",
                    parts=(
                        TextPart(
                            text=build_gen_soul_semantic_projection_user_prompt(
                                message_count=len(messages),
                                image_part_count=image_part_count,
                            )
                        ),
                    ),
                ),
            ),
            schema=SemanticProjectionPayload,
            temperature=0.1,
            timeout_s=self._timeout_s,
            metadata={"operation": "semantic_projection_v6"},
            max_retries=self._max_retries,
        )
        projected = normalize_text(payload.semantic_text)
        if not projected:
            raise ValueError("Semantic projection returned empty semantic_text")
        return projected

    def extract(
        self,
        messages: Sequence[Message],
        embedding: np.ndarray,
        schema: PsychologicalSchema,
        recent_tags: Optional[Sequence[str]] = None,
    ) -> MeaningExtraction:
        """深度语义提取：将文本映射到指定的动态 Schema 轴上"""
        semantic_text = self.project(messages)
        image_part_count = _image_part_count(messages)
        payload = self._llm.complete_json(
            messages=(
                Message(role="system", parts=(TextPart(text=GEN_SOUL_MEANING_SYSTEM_PROMPT),)),
                *messages,
                Message(
                    role="user",
                    parts=(
                        TextPart(
                            text=build_gen_soul_meaning_user_prompt(
                                text=semantic_text,
                                axis_names=schema.ordered_axis_names(),
                                recent_tags=recent_tags,
                                message_count=len(messages),
                                image_part_count=image_part_count,
                            )
                        ),
                    ),
                ),
            ),
            schema=MeaningFramePayloadV4,
            temperature=0.1,
            timeout_s=self._timeout_s,
            metadata={"operation": "meaning_extraction_v6"},
            max_retries=self._max_retries,
        )

        # 整理轴得分，确保符合当前的 Schema
        axis_evidence = {
            schema.canonical_axis_name(key): clamp(float(value))
            for key, value in payload.axis_evidence.items()
            if schema.canonical_axis_name(key) in schema.all_axes()
        }
        for axis_name in schema.ordered_axis_names():
            axis_evidence.setdefault(axis_name, 0.0)

        return MeaningExtraction(
            semantic_text=semantic_text,
            frame=EventFrame(
                axis_evidence=axis_evidence,
                valence=clamp(payload.valence),
                arousal=clamp01(payload.arousal),
                care=clamp01(payload.care),
                threat=clamp01(payload.threat),
                control=clamp01(payload.control),
                abandonment=clamp01(payload.abandonment),
                agency_signal=clamp01(payload.agency_signal),
                shame=clamp01(payload.shame),
                novelty=clamp01(payload.novelty),
                self_relevance=clamp01(payload.self_relevance),
                tags=tuple(payload.tags[:12]),
            ),
        )


class CombinatorialNarrativeProvider:
    """基于规则组合的叙事生成器。"""

    # 预定义的文学开场白和叙事短语
    OPENINGS = [
        "She tries to make the event legible to herself.",
        "She does not deny the feeling, but reorders its meaning.",
        "She lets the conflict stay long enough to become intelligible.",
        "She keeps the shock in view and starts rewriting the explanation.",
    ]
    DREAM_OPENERS = [
        "In the dream, time folds slightly.",
        "At the edge of sleep, unfinished fragments keep moving.",
        "Night does not replay reality; it tests a different path.",
    ]

    def compose_summary(
        self,
        state: IdentityState,
        schema: PsychologicalSchema,
        recent_semantic_texts: Sequence[str],
    ) -> NarrativeSummary:
        """生成身份摘要"""
        salient_axes = [
            name
            for name, _ in sorted(
                state.axis_state.items(), key=lambda item: abs(item[1]), reverse=True
            )[:4]
        ]
        axes_text = top_axes_description(state.axis_state, schema, topn=4)
        recent_hint = ""
        if recent_semantic_texts:
            snippet = recent_semantic_texts[-1][:48]
            recent_hint = f" Recent signal: {snippet}"
        text = (
            f"Current mode is {state.current_mode_label}. "
            f"Salient inner directions: {axes_text}. "
            f"Pressure={state.narrative_pressure():.2f}, "
            f"active={state.active_energy:.2f}, repressed={state.repressed_energy:.2f}."
            f"{recent_hint}"
        )
        return NarrativeSummary(
            text=text,
            current_mode=state.current_mode_label,
            pressure=state.narrative_pressure(),
            salient_axes=salient_axes,
        )

    def compose_repair(
        self,
        mode: str,
        state_before: IdentityState,
        state_after: IdentityState,
        dissonance_total: float,
        salient_axes: Sequence[str],
        plot_text: str,
        schema: PsychologicalSchema,
    ) -> str:
        """生成修复性叙事描述"""
        opening = random.choice(self.OPENINGS)
        axis_text = axes_to_phrase(salient_axes, schema)
        if mode == "preserve":
            body = "She tightens around what still feels load-bearing, refusing to let the shock rewrite everything."
        elif mode == "reframe":
            body = "She shifts the conflict from blunt fact into meaning, trying to keep continuity without denial."
        elif mode == "revise":
            body = "Reality pushes the same wound repeatedly, so she revises the older story of herself."
        elif mode == "differentiate":
            body = "She reclaims authorship and turns hurt into a boundary she can name."
        else:
            body = "She stops choosing between softness and defense, and lets both live in one narrative."
        snippet = plot_text[:56] + ("..." if len(plot_text) > 56 else "")
        return f"{opening} Faced with '{snippet}', the strain concentrates around {axis_text}. {body} Dissonance={dissonance_total:.2f}."

    def compose_dream(
        self,
        operator: str,
        fragment_tags: Sequence[str],
        state: IdentityState,
        schema: PsychologicalSchema,
    ) -> str:
        """生成梦境叙事描述"""
        opener = random.choice(self.DREAM_OPENERS)
        tags = ", ".join(fragment_tags[:4]) or "unfinished feelings"
        if operator == "counterfactual":
            body = (
                f"She reruns {tags} with a different boundary, testing whether the ending changes."
            )
        elif operator == "integration":
            body = f"{tags} are stitched into the same cloth until pain and meaning become visible together."
        elif operator == "fear_rehearsal":
            body = f"She rehearses the worst version of {tags}, searching for the last piece of agency."
        elif operator == "wish_rehearsal":
            body = f"She quietly rehearses a gentler version of {tags}, as if proximity might still hold."
        else:
            body = f"The fragments around {tags} keep circling until they start to interpret one another."
        return f"{opener} {body}"

    def label_mode(
        self,
        prototype_axes: Dict[str, float],
        schema: PsychologicalSchema,
        support: int,
    ) -> str:
        """为发现的模式生成人类可理解的标签"""
        return _bootstrap_mode_label(prototype_axes, schema, support)

    def bootstrap_mode_label(
        self,
        prototype_axes: Dict[str, float],
        schema: PsychologicalSchema,
        support: int,
    ) -> str:
        return self.label_mode(prototype_axes, schema, support)

    def judge_axis_merge(
        self,
        canonical: AxisSpec,
        alias: AxisSpec,
        evidence_overlap: Sequence[str],
    ) -> tuple[bool, str]:
        """基于语义重合度判断轴是否应当合并"""
        shared = set(tokenize_loose(" ".join(evidence_overlap)))
        score = 0.0
        if canonical.positive_pole.lower() == alias.positive_pole.lower():
            score += 1.0
        if canonical.negative_pole.lower() == alias.negative_pole.lower():
            score += 0.5
        if (
            canonical.name in alias.description.lower()
            or alias.name in canonical.description.lower()
        ):
            score += 0.5
        if shared:
            score += 0.2
        return score >= 1.2, "heuristic semantic overlap"


class LLMNarrativeProvider:
    """基于大模型的叙事生成器，提供富有文学性和连贯性的描述"""

    def __init__(
        self,
        llm: LLMProvider,
        *,
        timeout_s: float = 12.0,
        max_retries: int = 1,
    ) -> None:
        self._llm = llm
        self._timeout_s = timeout_s
        self._max_retries = max_retries

    def compose_summary(
        self,
        state: IdentityState,
        schema: PsychologicalSchema,
        recent_semantic_texts: Sequence[str],
    ) -> NarrativeSummary:
        """生成带有深度解读的自我摘要"""
        salient_axes = _salient_axis_names(state)
        payload = self._llm.complete_json(
            messages=_text_prompt_messages(
                system=GEN_SOUL_SUMMARY_SYSTEM_PROMPT,
                user=build_gen_soul_summary_user_prompt(
                    current_mode=state.current_mode_label,
                    salient_axes=salient_axes,
                    recent_semantic_texts=recent_semantic_texts,
                    pressure=state.narrative_pressure(),
                ),
            ),
            schema=NarrativeSummaryPayloadV4,
            temperature=0.2,
            timeout_s=self._timeout_s,
            metadata={"operation": "narrative_summary_v4"},
            max_retries=self._max_retries,
        )
        text = normalize_text(payload.text)
        if not text:
            raise ValueError("Narrative summary returned empty text")
        return NarrativeSummary(
            text=text,
            current_mode=normalize_text(payload.current_mode) or state.current_mode_label,
            pressure=state.narrative_pressure(),
            salient_axes=list(payload.salient_axes or salient_axes),
        )

    def compose_repair(
        self,
        mode: str,
        state_before: IdentityState,
        state_after: IdentityState,
        dissonance_total: float,
        salient_axes: Sequence[str],
        plot_text: str,
        schema: PsychologicalSchema,
    ) -> str:
        """生成深度的身份修复解释文案"""
        payload = self._llm.complete_json(
            messages=_text_prompt_messages(
                system=GEN_SOUL_REPAIR_SYSTEM_PROMPT,
                user=build_gen_soul_repair_user_prompt(
                    mode=mode,
                    plot_text=plot_text,
                    salient_axes=salient_axes,
                    dissonance_total=dissonance_total,
                ),
            ),
            schema=RepairNarrationPayloadV4,
            temperature=0.2,
            timeout_s=self._timeout_s,
            metadata={"operation": "repair_narration_v4"},
            max_retries=self._max_retries,
        )
        text = normalize_text(payload.text)
        if not text:
            raise ValueError("Repair narration returned empty text")
        return text

    def compose_dream(
        self,
        operator: str,
        fragment_tags: Sequence[str],
        state: IdentityState,
        schema: PsychologicalSchema,
    ) -> str:
        """生成充满意象的梦境文本"""
        payload = self._llm.complete_json(
            messages=_text_prompt_messages(
                system=GEN_SOUL_DREAM_SYSTEM_PROMPT,
                user=build_gen_soul_dream_user_prompt(
                    operator=operator, fragment_tags=fragment_tags
                ),
            ),
            schema=DreamNarrationPayloadV4,
            temperature=0.3,
            timeout_s=self._timeout_s,
            metadata={"operation": "dream_narration_v4"},
            max_retries=self._max_retries,
        )
        text = normalize_text(payload.text)
        if not text:
            raise ValueError("Dream narration returned empty text")
        return text

    def label_mode(
        self,
        prototype_axes: Dict[str, float],
        schema: PsychologicalSchema,
        support: int,
    ) -> str:
        """利用 LLM 为身份模式生成地道的文学化标签"""
        payload = self._llm.complete_json(
            messages=_text_prompt_messages(
                system=GEN_SOUL_MODE_LABEL_SYSTEM_PROMPT,
                user=build_gen_soul_mode_label_user_prompt(prototype_axes=prototype_axes),
            ),
            schema=ModeLabelPayloadV4,
            temperature=0.1,
            timeout_s=self._timeout_s,
            metadata={"operation": "mode_label_v4"},
            max_retries=self._max_retries,
        )
        label = normalize_text(payload.label)
        if not label:
            raise ValueError("Mode label returned empty label")
        return label

    def bootstrap_mode_label(
        self,
        prototype_axes: Dict[str, float],
        schema: PsychologicalSchema,
        support: int,
    ) -> str:
        return _bootstrap_mode_label(prototype_axes, schema, support)

    def judge_axis_merge(
        self,
        canonical: AxisSpec,
        alias: AxisSpec,
        evidence_overlap: Sequence[str],
    ) -> tuple[bool, str]:
        """利用 LLM 智能判断两个心理轴是否应当合并"""
        payload = self._llm.complete_json(
            messages=_text_prompt_messages(
                system=GEN_SOUL_AXIS_MERGE_SYSTEM_PROMPT,
                user=build_gen_soul_axis_merge_user_prompt(
                    canonical_name=canonical.name,
                    canonical_desc=canonical.description,
                    alias_name=alias.name,
                    alias_desc=alias.description,
                    evidence_overlap=evidence_overlap,
                ),
            ),
            schema=AxisMergeJudgementPayload,
            temperature=0.1,
            timeout_s=self._timeout_s,
            metadata={"operation": "axis_merge_judgement_v4"},
            max_retries=self._max_retries,
        )
        return bool(payload.should_merge), normalize_text(payload.rationale)
