"""
AURORA Generative Soul Memory
=============================

A hybrid memory / identity engine that removes persona-specific hardcoding
without removing the structural invariants required for stability.

Core stance
-----------
Do NOT remove all hardcoding. Split it into two kinds:

1) Load-bearing invariants (keep them):
   - bounded resources
   - factual vs synthetic memory separation
   - narrative homeostasis (active/repressed energy, contradiction EMA)
   - dream quarantine (dreams cannot directly rewrite facts)
   - phase transition hysteresis / refractory period
   - retrieval feedback shaping the metric / edge beliefs

2) Persona semantics (make them dynamic):
   - fixed traits -> dynamic persona axes discovered from profile + experience
   - fixed phases -> dynamic identity modes clustered from self-state history
   - fixed repair sentences -> generated narrative explanations
   - regex-only meaning extraction -> schema-driven evidence extraction with
     pluggable LLM hooks

This file is self-contained and runnable with:
    python aurora_generative_soul_memory.py

Dependencies:
    numpy
    networkx

Production notes
----------------
- Replace HashEmbedding with a real embedding model.
- Replace HeuristicMeaningProvider / CombinatorialNarrator with LLM JSON tools.
- Persist MemoryGraph / vector index / plot store in your backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, Literal
import copy
import hashlib
import json
import math
import random
import re
import time
import uuid

import networkx as nx
import numpy as np


# ============================================================================
# Utilities
# ============================================================================


def now_ts() -> float:
    return time.time()


def stable_hash(text: str) -> int:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def softmax(xs: Sequence[float], temperature: float = 1.0) -> List[float]:
    t = max(float(temperature), 1e-6)
    vals = [float(x) / t for x in xs]
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    Z = sum(exps) + 1e-12
    return [e / Z for e in exps]


def mean_abs(xs: Iterable[float]) -> float:
    arr = [abs(float(x)) for x in xs]
    return sum(arr) / max(len(arr), 1)


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v.astype(np.float32, copy=True)
    return (v / n).astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    aa = l2_normalize(a)
    bb = l2_normalize(b)
    return float(np.dot(aa, bb))


def moving_average(old: float, new: float, rate: float) -> float:
    return (1.0 - rate) * float(old) + rate * float(new)


def stable_uuid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def tokenize_loose(text: str) -> List[str]:
    # Works for mixed English / CJK reasonably enough for a fallback extractor.
    words = re.findall(r"[A-Za-z][A-Za-z_\-]{2,}|[\u4e00-\u9fff]{1,6}", text.lower())
    return [w for w in words if len(w.strip()) > 0]


def weighted_choice(
    items: Sequence[Any], weights: Sequence[float], rng: np.random.Generator
) -> Any:
    if not items:
        raise ValueError("weighted_choice on empty items")
    ws = np.array([max(0.0, float(w)) for w in weights], dtype=np.float64)
    if float(ws.sum()) <= 1e-12:
        idx = int(rng.integers(0, len(items)))
        return items[idx]
    ws = ws / ws.sum()
    idx = int(rng.choice(np.arange(len(items)), p=ws))
    return items[idx]


# ============================================================================
# Embedding
# ============================================================================


class HashEmbedding:
    """
    Deterministic bag-of-subtoken embedding.
    Replace with a real embedding model in production.
    """

    def __init__(self, dim: int = 384, seed: int = 0):
        self.dim = dim
        self.seed = seed

    def _vec_for_token(self, token: str) -> np.ndarray:
        h = stable_hash(f"{self.seed}:{token}")
        rng = np.random.default_rng(h % (2**32))
        v = rng.standard_normal(self.dim).astype(np.float32)
        return l2_normalize(v)

    def encode(self, text: str) -> np.ndarray:
        toks = tokenize_loose(text)
        if not toks:
            toks = [text[:32] or "empty"]
        acc = np.zeros(self.dim, dtype=np.float32)
        for t in toks:
            acc += self._vec_for_token(t)
        return l2_normalize(acc)


# ============================================================================
# Dynamic schema
# ============================================================================

ANTONYM_HINTS: Dict[str, str] = {
    "严谨": "草率",
    "rigor": "sloppiness",
    "严密": "松散",
    "冷酷": "温情",
    "scientific": "improvised",
    "科学": "迷信",
    "好奇": "麻木",
    "curious": "numb",
    "skeptical": "gullible",
    "怀疑": "轻信",
    "playful": "rigid",
    "跳脱": "刻板",
    "刺客": "守护者",
    "precise": "careless",
    "honest": "deceptive",
    "诚实": "欺瞒",
    "independent": "dependent",
    "独立": "依附",
    "protective": "exposed",
    "防御": "暴露",
    "柔软": "僵硬",
    "warm": "cold",
    "温柔": "冷淡",
    "creative": "formulaic",
    "创造": "公式化",
    "resilient": "fragile",
    "坚韧": "脆弱",
}

# Load-bearing homeostatic channels.
# These are NOT persona-specific traits; they are generic regulatory degrees of freedom.
HOMEOSTATIC_AXES: Tuple[Tuple[str, str, str, str], ...] = (
    ("affiliation", "靠近", "疏离", "Need for closeness vs distance"),
    ("agency", "主动", "被动", "Action / boundary / self-authorship"),
    ("exploration", "探索", "封闭", "Novelty seeking vs guarded closure"),
    ("vigilance", "警觉", "松弛", "Threat scanning vs trustful ease"),
    ("coherence", "整合", "碎裂", "Narrative integration vs fragmentation"),
    ("regulation", "稳态", "失控", "Affect regulation vs reactivity"),
)


@dataclass
class AxisSpec:
    name: str
    positive_pole: str
    negative_pole: str
    description: str
    level: Literal["homeostatic", "persona"] = "persona"
    weight: float = 1.0
    positive_examples: Tuple[str, ...] = ()
    negative_examples: Tuple[str, ...] = ()
    direction: Optional[np.ndarray] = None
    positive_anchor: Optional[np.ndarray] = None
    negative_anchor: Optional[np.ndarray] = None

    def compile(self, embedder: HashEmbedding) -> None:
        pos_text = (
            " ".join([self.name, self.positive_pole, self.description, *self.positive_examples])
            or self.positive_pole
        )
        neg_text = (
            " ".join([self.name, self.negative_pole, self.description, *self.negative_examples])
            or self.negative_pole
        )
        self.positive_anchor = embedder.encode(pos_text)
        self.negative_anchor = embedder.encode(neg_text)
        self.direction = l2_normalize(self.positive_anchor - self.negative_anchor)

    def score(self, text_embedding: np.ndarray, text: str = "") -> float:
        if self.direction is None or self.positive_anchor is None or self.negative_anchor is None:
            raise ValueError(f"Axis {self.name} not compiled")
        sim_pos = cosine_sim(text_embedding, self.positive_anchor)
        sim_neg = cosine_sim(text_embedding, self.negative_anchor)
        base = clamp(sim_pos - sim_neg)
        # Keyword anchoring boosts interpretability for the fallback implementation.
        t = text.lower()
        for w in [self.positive_pole, *self.positive_examples]:
            if w and w.lower() in t:
                base += 0.18
        for w in [self.negative_pole, *self.negative_examples]:
            if w and w.lower() in t:
                base -= 0.18
        return clamp(base)


@dataclass
class PsychologicalSchema:
    homeostatic_axes: Dict[str, AxisSpec] = field(default_factory=dict)
    persona_axes: Dict[str, AxisSpec] = field(default_factory=dict)
    profile_text: str = ""

    def all_axes(self) -> Dict[str, AxisSpec]:
        merged = dict(self.homeostatic_axes)
        merged.update(self.persona_axes)
        return merged

    def ordered_axis_names(self) -> List[str]:
        return list(self.homeostatic_axes.keys()) + list(self.persona_axes.keys())

    def compile(self, embedder: HashEmbedding) -> None:
        for ax in self.all_axes().values():
            ax.compile(embedder)

    def add_persona_axis(self, axis: AxisSpec, embedder: HashEmbedding) -> None:
        axis.compile(embedder)
        self.persona_axes[axis.name] = axis

    @classmethod
    def from_profile(
        cls,
        embedder: HashEmbedding,
        profile_text: str = "",
        persona_axes: Optional[List[Dict[str, Any]]] = None,
    ) -> "PsychologicalSchema":
        schema = cls(profile_text=profile_text)
        for name, pos, neg, desc in HOMEOSTATIC_AXES:
            schema.homeostatic_axes[name] = AxisSpec(
                name=name,
                positive_pole=pos,
                negative_pole=neg,
                description=desc,
                level="homeostatic",
            )

        built: List[AxisSpec] = []
        if persona_axes:
            for item in persona_axes:
                built.append(
                    AxisSpec(
                        name=str(item.get("name")),
                        positive_pole=str(item.get("positive_pole", item.get("positive", "more"))),
                        negative_pole=str(item.get("negative_pole", item.get("negative", "less"))),
                        description=str(item.get("description", "")),
                        level="persona",
                        positive_examples=tuple(item.get("positive_examples", []) or []),
                        negative_examples=tuple(item.get("negative_examples", []) or []),
                        weight=float(item.get("weight", 1.0)),
                    )
                )
        else:
            built.extend(_persona_axes_from_profile_text(profile_text))

        for axis in built:
            schema.persona_axes[axis.name] = axis

        schema.compile(embedder)
        return schema


def _persona_axes_from_profile_text(profile_text: str) -> List[AxisSpec]:
    """
    Parse free text profile into persona axes.

    Examples accepted:
    - "严谨的科学家，好奇但怀疑"
    - "cold assassin; precise; detached"
    - JSON-like pairs can be passed through persona_axes instead
    """
    text = profile_text.strip()
    if not text:
        return []

    chunks = re.split(r"[,\n;；，|、]+", text)
    axes: List[AxisSpec] = []
    seen: set[str] = set()

    def _simplify(fragment: str) -> str:
        frag = fragment.strip()
        if not frag:
            return frag
        # Prefer substrings that match known antonym hints.
        for hint in sorted(ANTONYM_HINTS.keys(), key=len, reverse=True):
            if hint in frag:
                return hint
        # Strip common function words.
        frag = re.sub(r"^(很|更|太|比较|一个|会|会变得|变得|想要|重视)", "", frag)
        frag = re.sub(r"(的人|的人格|的角色|的方式)$", "", frag)
        if "的" in frag and len(frag.split("的")[0]) >= 2:
            frag = frag.split("的")[0]
        return frag[:12]

    for idx, raw in enumerate(chunks):
        for chunk0 in re.split(r"\bbut\b|\band\b|但|又|以及|和|与|\+", raw, flags=re.IGNORECASE):
            chunk = chunk0.strip()
            if not chunk:
                continue

            # pattern 1: "X/Y" or "X vs Y"
            m = re.search(r"(.+?)(?:/| vs |↔|->|=>| to )(.+)", chunk, flags=re.IGNORECASE)
            if m:
                pos = _simplify(m.group(1).strip())
                neg = _simplify(m.group(2).strip())
                name = re.sub(r"\W+", "_", pos.lower())[:24] or f"persona_{idx}"
                if name in seen:
                    continue
                seen.add(name)
                axes.append(
                    AxisSpec(
                        name=name,
                        positive_pole=pos,
                        negative_pole=neg,
                        description=f"Persona dimension derived from profile fragment: {chunk}",
                        level="persona",
                    )
                )
                continue

            words = tokenize_loose(chunk)
            if not words:
                continue
            head = _simplify(chunk) or words[0]
            neg = ANTONYM_HINTS.get(head, f"not_{head}")
            name = re.sub(r"\W+", "_", head.lower())[:24] or f"persona_{idx}"
            if name in seen:
                continue
            seen.add(name)
            axes.append(
                AxisSpec(
                    name=name,
                    positive_pole=head,
                    negative_pole=neg,
                    description=f"Persona dimension derived from profile fragment: {chunk}",
                    level="persona",
                    positive_examples=tuple(words[:4]),
                )
            )

    return axes


# ============================================================================
# Event frames & narrative providers
# ============================================================================


@dataclass
class EventFrame:
    axis_evidence: Dict[str, float] = field(default_factory=dict)  # signed [-1, 1]
    valence: float = 0.0
    arousal: float = 0.0
    care: float = 0.0
    threat: float = 0.0
    control: float = 0.0
    abandonment: float = 0.0
    agency_signal: float = 0.0
    shame: float = 0.0
    novelty: float = 0.0
    self_relevance: float = 0.5
    tags: Tuple[str, ...] = ()

    def evidence_strength(self) -> float:
        return (
            0.45 * mean_abs(self.axis_evidence.values())
            + 0.20 * abs(self.valence)
            + 0.20 * self.arousal
            + 0.15 * self.self_relevance
        )


class MeaningProvider(Protocol):
    def extract(
        self,
        text: str,
        embedding: np.ndarray,
        schema: PsychologicalSchema,
        recent_tags: Optional[Sequence[str]] = None,
    ) -> EventFrame: ...


class NarrativeProvider(Protocol):
    def compose_summary(
        self,
        state: "IdentityState",
        schema: PsychologicalSchema,
        recent_texts: Sequence[str],
    ) -> str: ...

    def compose_repair(
        self,
        mode: str,
        state_before: "IdentityState",
        state_after: "IdentityState",
        dissonance: "DissonanceReport",
        salient_axes: Sequence[str],
        plot: "Plot",
        schema: PsychologicalSchema,
    ) -> str: ...

    def compose_dream(
        self,
        operator: str,
        fragments: Sequence["LatentFragment"],
        state: "IdentityState",
        schema: PsychologicalSchema,
    ) -> str: ...

    def label_mode(
        self,
        prototype_axes: Dict[str, float],
        schema: PsychologicalSchema,
        support: int,
    ) -> str: ...


class HeuristicMeaningProvider:
    """
    Schema-driven fallback extractor.
    It is more generic than fixed-regex trait extraction because the axes themselves
    are dynamic and compiled from profile / discovered schema.
    """

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
        "safe",
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

    def extract(
        self,
        text: str,
        embedding: np.ndarray,
        schema: PsychologicalSchema,
        recent_tags: Optional[Sequence[str]] = None,
    ) -> EventFrame:
        t = text.lower()
        toks = set(tokenize_loose(text))

        axis_evidence: Dict[str, float] = {}
        for name, axis in schema.all_axes().items():
            axis_evidence[name] = axis.score(embedding, text)

        care = self._lex_score(t, toks, self.CARE_WORDS)
        threat = self._lex_score(t, toks, self.THREAT_WORDS)
        shame = self._lex_score(t, toks, self.SHAME_WORDS)
        agency_signal = self._lex_score(t, toks, self.AGENCY_WORDS)
        control = self._lex_score(t, toks, self.CONTROL_WORDS)
        abandonment = self._lex_score(t, toks, self.ABANDON_WORDS)

        pos = self._lex_score(t, toks, self.POSITIVE_WORDS)
        neg = self._lex_score(t, toks, self.NEGATIVE_WORDS)
        valence = clamp(pos - neg)
        arousal = clamp01(0.25 + 0.45 * threat + 0.20 * shame + 0.15 * agency_signal + 0.15 * neg)

        self_relevance = 0.40
        if any(x in t for x in ["我", "自己", "她", "你", "we", "i ", " me ", " my ", "you "]):
            self_relevance += 0.18
        if "self" in t:
            self_relevance += 0.15
        self_relevance = clamp01(self_relevance)

        tags = self._make_tags(
            text,
            axis_evidence,
            care,
            threat,
            shame,
            agency_signal,
            control,
            abandonment,
            recent_tags,
        )
        novelty = clamp01(
            0.15
            + 0.55
            * len([k for k, v in axis_evidence.items() if abs(v) > 0.25])
            / max(len(axis_evidence), 1)
        )
        novelty = clamp01(novelty + 0.12 * max(0, len(set(tags)) - 3))

        return EventFrame(
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
        )

    def _lex_score(self, text: str, toks: set[str], lex: set[str]) -> float:
        hits = 0
        for w in lex:
            if w in text or w.lower() in toks:
                hits += 1
        if hits == 0:
            return 0.0
        return clamp01(0.22 * hits)

    def _make_tags(
        self,
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
        tags: List[str] = []
        for name, v in sorted(axis_evidence.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]:
            if abs(v) > 0.18:
                tags.append(name)
                tags.append(f"{name}:{'pos' if v >= 0 else 'neg'}")
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

        toks = tokenize_loose(text)
        content_tags = [tok for tok in toks if len(tok) >= 2][:6]
        for tok in content_tags[:4]:
            if tok not in tags:
                tags.append(tok)
        if recent_tags:
            for rt in recent_tags[-4:]:
                if rt in tags:
                    tags.append(f"echo:{rt}")
        return tags[:12]


class CombinatorialNarrator:
    """
    Runnable fallback "generative" narrator.
    This is not a true LLM, but it avoids static one-line templates by combining
    state, salient axes, fragments, and multiple lexical families.
    """

    OPENINGS = [
        "她试着把这件事讲通",
        "她开始重新安排自己的解释",
        "她没有直接否认感受，而是重新整理",
        "她把冲突留在心里，反复推演后才开口",
        "她在叙事里给自己留出了一条新的路",
    ]
    DEFENSE_VERBS = ["挡住", "收紧", "后退", "把门关小一点", "把自己藏好一点"]
    REFRAME_VERBS = ["换个角度看", "重新命名", "改写解释", "把痛感翻译成含义", "让碎片重新排序"]
    REVISE_VERBS = [
        "承认改变",
        "移动自我边界",
        "修订旧版本的自己",
        "把旧叙事拆开重写",
        "接受自己已经不同",
    ]
    DIFFERENTIATE_VERBS = [
        "立起边界",
        "把主动权拿回来",
        "把伤口改造成规则",
        "停止无条件让渡",
        "学会带着清醒靠近",
    ]
    INTEGRATE_VERBS = [
        "把矛盾容纳进来",
        "允许柔软和边界并存",
        "把旧伤与新证据放在同一张桌子上",
        "让自己更完整",
        "把裂缝缝合成纹理",
    ]

    DREAM_OPENERS = [
        "梦里，时间被轻轻折了一下。",
        "在梦的边缘，她把白天没说完的话继续下去。",
        "潜意识没有照抄现实，而是偷偷试了一种别的走法。",
        "夜里，那些碎片开始自动寻找彼此。",
    ]

    def compose_summary(
        self,
        state: "IdentityState",
        schema: PsychologicalSchema,
        recent_texts: Sequence[str],
    ) -> str:
        axes = _top_axes_description(state.axis_state, schema, topn=4)
        mode = state.current_mode_label or "未命名模式"
        recent_hint = ""
        if recent_texts:
            snippet = recent_texts[-1]
            snippet = snippet[:36] + ("..." if len(snippet) > 36 else "")
            recent_hint = f" 最近的震动仍来自：{snippet}"
        return (
            f"当前她以「{mode}」的方式维持自我。"
            f"最显著的内在方向是：{axes}。"
            f"稳态压力={state.narrative_pressure():.2f}，"
            f"活动能量={state.active_energy:.2f}，压抑能量={state.repressed_energy:.2f}。"
            f"{recent_hint}"
        )

    def compose_repair(
        self,
        mode: str,
        state_before: "IdentityState",
        state_after: "IdentityState",
        dissonance: "DissonanceReport",
        salient_axes: Sequence[str],
        plot: "Plot",
        schema: PsychologicalSchema,
    ) -> str:
        op = random.choice(self.OPENINGS)
        axis_desc = _axes_to_phrase(salient_axes, schema)
        if mode == "preserve":
            verb = random.choice(self.DEFENSE_VERBS)
            core = f"她先选择{verb}，不让这次冲击直接改写自己。"
        elif mode == "reframe":
            verb = random.choice(self.REFRAME_VERBS)
            core = f"她试着{verb}，把矛盾从事实判断转成意义判断。"
        elif mode == "revise":
            verb = random.choice(self.REVISE_VERBS)
            core = f"现实反复冲撞同一处，她只能{verb}。"
        elif mode == "differentiate":
            verb = random.choice(self.DIFFERENTIATE_VERBS)
            core = f"这一次她决定{verb}。"
        else:
            verb = random.choice(self.INTEGRATE_VERBS)
            core = f"她没有再做二选一，而是{verb}。"

        pressure_part = f"冲突总量={dissonance.total:.2f}"
        text_hint = plot.text[:34] + ("..." if len(plot.text) > 34 else "")
        return f"{op} 面对「{text_hint}」，核心牵动在 {axis_desc}。{core} {pressure_part}。"

    def compose_dream(
        self,
        operator: str,
        fragments: Sequence["LatentFragment"],
        state: "IdentityState",
        schema: PsychologicalSchema,
    ) -> str:
        opener = random.choice(self.DREAM_OPENERS)
        tags: List[str] = []
        for frag in fragments:
            for t in frag.tags:
                if t not in tags:
                    tags.append(t)
        tags = tags[:4]
        tag_phrase = "、".join(tags) if tags else "几段没有完成的情绪"

        if operator == "counterfactual":
            body = f"她把 {tag_phrase} 倒过来演了一遍，看看如果自己早一点说出边界，结局会不会不同。"
        elif operator == "integration":
            body = f"{tag_phrase} 在梦里被缝成同一块布，她第一次同时看见疼痛和意义。"
        elif operator == "fear_rehearsal":
            body = "她预演最坏的版本：如果一切再次失控，自己还能怎样留下主动权。"
        elif operator == "wish_rehearsal":
            body = "潜意识偷偷试了一次更温柔的路线，像是在预演一种尚未发生但可能成立的靠近。"
        else:
            body = "那些碎片围着她转，直到它们在某个位置忽然彼此解释了起来。"
        return f"{opener} {body}"

    def label_mode(
        self,
        prototype_axes: Dict[str, float],
        schema: PsychologicalSchema,
        support: int,
    ) -> str:
        items = sorted(prototype_axes.items(), key=lambda kv: abs(kv[1]), reverse=True)
        words: List[str] = []
        for name, value in items[:2]:
            ax = schema.all_axes().get(name)
            if not ax:
                continue
            pole = ax.positive_pole if value >= 0 else ax.negative_pole
            words.append(pole)
        if not words:
            return f"mode_{support}"
        if len(words) == 1:
            return f"{words[0]}模式"
        return f"{words[0]}—{words[1]}模式"


# ============================================================================
# Memory primitives
# ============================================================================


@dataclass
class Plot:
    id: str
    ts: float
    text: str
    actors: Tuple[str, ...]
    embedding: np.ndarray
    frame: EventFrame

    source: Literal["wake", "dream", "repair", "mode"] = "wake"
    confidence: float = 1.0
    evidence_weight: float = 1.0

    surprise: float = 0.0
    redundancy: float = 0.0
    contradiction: float = 0.0
    tension: float = 0.0

    story_id: Optional[str] = None
    theme_id: Optional[str] = None

    access_count: int = 0
    last_access_ts: float = field(default_factory=now_ts)
    status: Literal["active", "absorbed", "archived"] = "active"

    def mass(self) -> float:
        age = max(1.0, now_ts() - self.ts)
        freshness = 1.0 / math.log1p(age)
        return freshness * math.log1p(self.access_count + 1.0) * (0.5 + 0.5 * self.confidence)


@dataclass
class StoryArc:
    id: str
    created_ts: float
    updated_ts: float
    plot_ids: List[str] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    actor_counts: Dict[str, int] = field(default_factory=dict)
    tag_counts: Dict[str, int] = field(default_factory=dict)
    source_counts: Dict[str, int] = field(default_factory=dict)
    tension_curve: List[float] = field(default_factory=list)
    reference_count: int = 0
    dist_mean: float = 0.0
    dist_m2: float = 0.0
    dist_n: int = 0
    gap_mean: float = 0.0
    gap_m2: float = 0.0
    gap_n: int = 0

    def mass(self) -> float:
        age = max(1.0, now_ts() - self.updated_ts)
        freshness = 1.0 / math.log1p(age)
        size = math.log1p(len(self.plot_ids) + 1)
        return freshness * size * (1.0 + 0.15 * self.reference_count)

    def _update_stat(self, kind: str, x: float) -> None:
        if kind == "dist":
            self.dist_n += 1
            delta = x - self.dist_mean
            self.dist_mean += delta / self.dist_n
            self.dist_m2 += delta * (x - self.dist_mean)
            return
        if kind == "gap":
            self.gap_n += 1
            delta = x - self.gap_mean
            self.gap_mean += delta / self.gap_n
            self.gap_m2 += delta * (x - self.gap_mean)
            return
        raise ValueError(kind)

    def dist_var(self) -> float:
        return self.dist_m2 / (self.dist_n - 1) if self.dist_n > 1 else 1.0

    def gap_mean_safe(self, default: float = 3600.0) -> float:
        return self.gap_mean if self.gap_n > 0 and self.gap_mean > 0 else default


@dataclass
class Theme:
    id: str
    created_ts: float
    updated_ts: float
    story_ids: List[str] = field(default_factory=list)
    prototype: Optional[np.ndarray] = None
    a: float = 1.0
    b: float = 1.0
    label: str = ""
    description: str = ""

    def confidence(self) -> float:
        return self.a / (self.a + self.b)

    def mass(self) -> float:
        age = max(1.0, now_ts() - self.updated_ts)
        freshness = 1.0 / math.log1p(age)
        return freshness * math.log1p(len(self.story_ids) + 1.0) * self.confidence()

    def update_evidence(self, success: bool) -> None:
        if success:
            self.a += 1.0
        else:
            self.b += 1.0
        self.updated_ts = now_ts()


@dataclass
class EdgeBelief:
    edge_type: str
    a: float = 1.0
    b: float = 1.0
    use_count: int = 0
    last_used_ts: float = field(default_factory=now_ts)

    def mean(self) -> float:
        return self.a / (self.a + self.b)

    def update(self, success: bool) -> None:
        self.use_count += 1
        self.last_used_ts = now_ts()
        if success:
            self.a += 1.0
        else:
            self.b += 1.0


class MemoryGraph:
    def __init__(self) -> None:
        self.g = nx.DiGraph()

    def add_node(self, node_id: str, kind: str, payload: Any) -> None:
        self.g.add_node(node_id, kind=kind, payload=payload)

    def payload(self, node_id: str) -> Any:
        return self.g.nodes[node_id]["payload"]

    def kind(self, node_id: str) -> str:
        return self.g.nodes[node_id]["kind"]

    def ensure_edge(self, src: str, dst: str, edge_type: str) -> None:
        if not self.g.has_edge(src, dst):
            self.g.add_edge(src, dst, belief=EdgeBelief(edge_type=edge_type))

    def edge_belief(self, src: str, dst: str) -> EdgeBelief:
        return self.g.edges[src, dst]["belief"]

    def nodes_of_kind(self, kind: str) -> List[str]:
        return [n for n, d in self.g.nodes(data=True) if d.get("kind") == kind]


class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.ids: List[str] = []
        self.kinds: List[str] = []
        self.vecs: List[np.ndarray] = []

    def add(self, node_id: str, vec: np.ndarray, kind: str) -> None:
        vec = vec.astype(np.float32)
        if vec.shape != (self.dim,):
            raise ValueError(f"Vector dim mismatch {vec.shape} != {(self.dim,)}")
        self.ids.append(node_id)
        self.kinds.append(kind)
        self.vecs.append(vec)

    def update(self, node_id: str, vec: np.ndarray) -> None:
        if node_id not in self.ids:
            self.add(node_id, vec, "unknown")
            return
        idx = self.ids.index(node_id)
        self.vecs[idx] = vec.astype(np.float32)

    def search(
        self, q: np.ndarray, k: int = 10, kind: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        hits: List[Tuple[str, float]] = []
        for node_id, kd, vec in zip(self.ids, self.kinds, self.vecs):
            if kind is not None and kd != kind:
                continue
            hits.append((node_id, cosine_sim(q, vec)))
        hits.sort(key=lambda x: x[1], reverse=True)
        return hits[:k]


# ============================================================================
# Online statistics
# ============================================================================


class OnlineKDE:
    def __init__(self, dim: int, reservoir: int = 1024, bandwidth: float = 0.85, seed: int = 0):
        self.dim = dim
        self.reservoir = reservoir
        self.bandwidth = bandwidth
        self.rng = np.random.default_rng(seed)
        self.samples: List[np.ndarray] = []

    def add(self, x: np.ndarray) -> None:
        x = x.astype(np.float32)
        if len(self.samples) < self.reservoir:
            self.samples.append(x)
            return
        i = int(self.rng.integers(0, len(self.samples) + 1))
        if i < self.reservoir:
            self.samples[i] = x

    def logprob(self, x: np.ndarray) -> float:
        if not self.samples:
            return -2.0
        vals = []
        h2 = self.bandwidth**2
        for s in self.samples[: min(len(self.samples), 256)]:
            d2 = float(np.sum((x - s) ** 2))
            vals.append(-0.5 * d2 / max(h2, 1e-6))
        m = max(vals)
        z = sum(math.exp(v - m) for v in vals) + 1e-12
        return m + math.log(z) - math.log(len(vals))

    def surprise(self, x: np.ndarray) -> float:
        return max(0.0, -self.logprob(x))


class LowRankMetric:
    def __init__(self, dim: int, rank: int = 32, seed: int = 0):
        self.dim = dim
        self.rank = rank
        rng = np.random.default_rng(seed)
        self.L = (0.08 * rng.standard_normal((rank, dim))).astype(np.float32)
        for i in range(min(rank, dim)):
            self.L[i, i] += 1.0

    def d2(self, a: np.ndarray, b: np.ndarray) -> float:
        z = self.L @ (a - b)
        return float(np.dot(z, z))

    def sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return math.exp(-0.5 * self.d2(a, b))

    def update_triplet(
        self,
        q: np.ndarray,
        pos: np.ndarray,
        neg: np.ndarray,
        lr: float = 0.01,
        margin: float = 0.25,
    ) -> None:
        dq = q - pos
        dn = q - neg
        sp = self.L @ dq
        sn = self.L @ dn
        loss = margin + float(np.dot(sp, sp)) - float(np.dot(sn, sn))
        if loss <= 0:
            return
        grad = 2.0 * (np.outer(sp, dq) - np.outer(sn, dn))
        self.L = (self.L - lr * grad).astype(np.float32)


class ThompsonEncodeGate:
    """
    A tiny stochastic encode gate.
    Not persona-specific; it decides whether storing a plot is worth the cost.
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.a: Dict[str, float] = {
            "surprise": 1.0,
            "tension": 1.0,
            "contradiction": 1.0,
            "arousal": 1.0,
            "novelty": 1.0,
            "dream": 1.0,
            "repair": 1.0,
            "wake": 1.0,
        }
        self.b: Dict[str, float] = {k: 1.0 for k in self.a}

    def should_encode(self, features: Dict[str, float], source: str) -> bool:
        score = 0.0
        total = 0.0
        for k, v in features.items():
            if k not in self.a:
                continue
            w = float(self.rng.beta(self.a[k], self.b[k]))
            score += w * clamp01(v)
            total += 1.0
        if source in self.a:
            score += float(self.rng.beta(self.a[source], self.b[source]))
            total += 1.0
        if total <= 0:
            return True
        p = clamp01(score / total)
        return bool(self.rng.random() < p)

    def update(self, source: str, features: Dict[str, float], success: bool) -> None:
        keys = [k for k in features.keys() if k in self.a]
        if source in self.a:
            keys.append(source)
        for k in keys:
            if success:
                self.a[k] += 0.20
            else:
                self.b[k] += 0.20


# ============================================================================
# Story / Theme assignment
# ============================================================================


class CRPAssigner:
    def __init__(self, alpha: float = 1.0, seed: int = 0):
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)

    def sample(self, logps: Dict[str, float]) -> Tuple[Optional[str], Dict[str, float]]:
        logs = dict(logps)
        logs["__new__"] = math.log(self.alpha)
        keys = list(logs.keys())
        probs = softmax([logs[k] for k in keys], temperature=1.0)
        choice = self.rng.choice(keys, p=np.array(probs, dtype=np.float64))
        post = {k: p for k, p in zip(keys, probs)}
        if choice == "__new__":
            return None, post
        return str(choice), post


class StoryModel:
    def __init__(self, metric: LowRankMetric):
        self.metric = metric

    def loglik(self, plot: Plot, story: StoryArc, plots: Dict[str, Plot]) -> float:
        ll_sem = 0.0
        if story.centroid is not None:
            d2 = self.metric.d2(plot.embedding, story.centroid)
            var = max(story.dist_var(), 1e-3)
            ll_sem = -0.5 * d2 / var

        ll_time = 0.0
        if story.plot_ids:
            gap = max(0.0, plot.ts - story.updated_ts)
            tau = story.gap_mean_safe()
            lam = 1.0 / max(tau, 1e-6)
            ll_time = math.log(lam + 1e-12) - lam * gap

        ll_actor = 0.0
        total = sum(story.actor_counts.values())
        beta = 1.0
        denom = total + beta * max(len(story.actor_counts), 1)
        for a in plot.actors:
            ll_actor += math.log(story.actor_counts.get(a, 0) + beta) - math.log(denom + 1e-12)

        ll_source = 0.0
        if story.source_counts:
            total_s = sum(story.source_counts.values())
            ll_source = math.log(story.source_counts.get(plot.source, 0) + 1.0) - math.log(
                total_s + len(story.source_counts) + 1e-12
            )

        ll_tag = 0.0
        tag_total = sum(story.tag_counts.values())
        if tag_total > 0:
            denom_t = tag_total + max(len(story.tag_counts), 1)
            for tg in plot.frame.tags[:4]:
                ll_tag += math.log(story.tag_counts.get(tg, 0) + 1.0) - math.log(denom_t + 1e-12)

        return ll_sem + ll_time + ll_actor + ll_source + ll_tag


class ThemeModel:
    def __init__(self, metric: LowRankMetric):
        self.metric = metric

    def loglik(self, story: StoryArc, theme: Theme) -> float:
        if theme.prototype is None or story.centroid is None:
            return 0.0
        d2 = self.metric.d2(story.centroid, theme.prototype)
        return -0.5 * d2


# ============================================================================
# Retrieval
# ============================================================================


@dataclass
class RetrievalTrace:
    query_text: str
    query_vec: np.ndarray
    attractor_vec: np.ndarray
    anchor_ids: List[str]
    ranked: List[Tuple[str, float]]
    evidence: List[str]


class FieldRetriever:
    def __init__(self, metric: LowRankMetric, vindex: VectorIndex, graph: MemoryGraph):
        self.metric = metric
        self.vindex = vindex
        self.graph = graph

    def _node_vector(self, node_id: str) -> np.ndarray:
        kind = self.graph.kind(node_id)
        payload = self.graph.payload(node_id)
        if kind == "plot":
            return payload.embedding
        if kind == "story":
            return (
                payload.centroid
                if payload.centroid is not None
                else np.zeros(self.vindex.dim, dtype=np.float32)
            )
        if kind == "theme":
            return (
                payload.prototype
                if payload.prototype is not None
                else np.zeros(self.vindex.dim, dtype=np.float32)
            )
        raise ValueError(kind)

    def attractor_trace(
        self, q: np.ndarray, steps: int = 3, k: int = 12
    ) -> Tuple[np.ndarray, List[str]]:
        cur = q.astype(np.float32)
        anchors: List[str] = []
        for _ in range(steps):
            hits = self.vindex.search(cur, k=k)
            if not hits:
                break
            vecs = []
            weights = []
            anchors = [hid for hid, _ in hits[:5]]
            for hid, sim in hits:
                payload = self.graph.payload(hid)
                vec = self._node_vector(hid)
                mass = payload.mass() if hasattr(payload, "mass") else 1.0
                w = max(0.0, sim) ** 2 * (0.5 + 0.5 * min(2.5, float(mass)))
                vecs.append(vec)
                weights.append(w)
            if sum(weights) <= 1e-9:
                break
            acc = np.zeros_like(cur)
            Z = sum(weights) + 1e-12
            for vec, w in zip(vecs, weights):
                acc += (w / Z) * vec
            cur = l2_normalize(acc)
        return cur, anchors

    def diffuse(
        self, anchor_ids: Sequence[str], alpha: float = 0.85, k: int = 10
    ) -> List[Tuple[str, float]]:
        if not anchor_ids:
            return []
        nodes = set(anchor_ids)
        for aid in anchor_ids:
            if aid not in self.graph.g:
                continue
            nodes.update(self.graph.g.predecessors(aid))
            nodes.update(self.graph.g.successors(aid))
            for n in list(nodes):
                if n in self.graph.g:
                    nodes.update(self.graph.g.predecessors(n))
                    nodes.update(self.graph.g.successors(n))
        sub = self.graph.g.subgraph(nodes).copy()
        if sub.number_of_nodes() == 0:
            return []

        personalization = {n: 1e-6 for n in sub.nodes}
        for aid in anchor_ids:
            if aid in personalization:
                personalization[aid] = 1.0
        s = sum(personalization.values()) + 1e-12
        personalization = {k_: v / s for k_, v in personalization.items()}

        scores = nx.pagerank(sub, alpha=alpha, personalization=personalization, max_iter=100)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:k]

    def query(self, text: str, qvec: np.ndarray, k: int = 8) -> RetrievalTrace:
        attractor, anchors = self.attractor_trace(qvec)
        coarse = self.vindex.search(attractor, k=max(k * 2, 12))
        diffused = self.diffuse([hid for hid, _ in coarse[:5]], k=max(k * 2, 12))

        fused: Dict[str, float] = {}
        for hid, sim in coarse:
            fused[hid] = fused.get(hid, 0.0) + 0.65 * sim
        for hid, pr in diffused:
            fused[hid] = fused.get(hid, 0.0) + 0.35 * pr

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        evidence: List[str] = []
        for hid, score in ranked[:5]:
            kind = self.graph.kind(hid)
            payload = self.graph.payload(hid)
            if kind == "plot":
                snippet = payload.text[:48] + ("..." if len(payload.text) > 48 else "")
                evidence.append(f"[plot {hid}] {snippet} (score={score:.3f})")
            elif kind == "story":
                evidence.append(f"[story {hid}] size={len(payload.plot_ids)} (score={score:.3f})")
            else:
                lbl = payload.label or hid
                evidence.append(f"[theme {hid}] {lbl} (score={score:.3f})")

        return RetrievalTrace(
            query_text=text,
            query_vec=qvec,
            attractor_vec=attractor,
            anchor_ids=anchors,
            ranked=ranked,
            evidence=evidence,
        )


# ============================================================================
# Identity / dissonance / dynamic modes
# ============================================================================


@dataclass
class DissonanceReport:
    axis_conflicts: Dict[str, float]
    axis_alignments: Dict[str, float]
    semantic_conflict: float
    affective_load: float
    narrative_incongruity: float
    total: float


@dataclass
class IdentityMode:
    id: str
    label: str
    prototype: np.ndarray
    axis_prototype: Dict[str, float]
    support: int = 1
    barrier: float = 0.55
    hysteresis: float = 0.08
    created_ts: float = field(default_factory=now_ts)
    updated_ts: float = field(default_factory=now_ts)

    def score(self, self_vec: np.ndarray, axis_state: Dict[str, float]) -> float:
        sem = cosine_sim(self_vec, self.prototype)
        ax = 0.0
        if self.axis_prototype:
            common = [k for k in self.axis_prototype.keys() if k in axis_state]
            if common:
                ax = np.mean([1.0 - abs(axis_state[k] - self.axis_prototype[k]) for k in common])
        return 0.65 * sem + 0.35 * float(ax)


@dataclass
class IdentityState:
    self_vector: np.ndarray
    axis_state: Dict[str, float] = field(default_factory=dict)
    intuition_axes: Dict[str, float] = field(default_factory=dict)
    active_energy: float = 0.0
    repressed_energy: float = 0.0
    contradiction_ema: float = 0.0
    summary: str = ""
    current_mode_id: Optional[str] = None
    current_mode_label: str = "origin"
    repair_count: int = 0
    dream_count: int = 0
    mode_change_count: int = 0
    last_mode_step: int = 0
    last_mode_change_ts: float = field(default_factory=now_ts)
    narrative_log: List[str] = field(default_factory=list)

    def plasticity(self) -> float:
        openness = (self.axis_state.get("exploration", 0.0) + 1.0) / 2.0
        coherence = (self.axis_state.get("coherence", 0.0) + 1.0) / 2.0
        regulation = (self.axis_state.get("regulation", 0.0) + 1.0) / 2.0
        vigilance = (self.axis_state.get("vigilance", 0.0) + 1.0) / 2.0
        return clamp01(
            0.35 * openness + 0.35 * coherence + 0.20 * regulation + 0.10 * (1.0 - vigilance)
        )

    def rigidity(self) -> float:
        openness = (self.axis_state.get("exploration", 0.0) + 1.0) / 2.0
        vigilance = (self.axis_state.get("vigilance", 0.0) + 1.0) / 2.0
        regulation = (self.axis_state.get("regulation", 0.0) + 1.0) / 2.0
        return clamp01(0.45 * vigilance + 0.35 * (1.0 - openness) + 0.20 * (1.0 - regulation))

    def narrative_pressure(self) -> float:
        return float(
            self.active_energy + 0.85 * self.repressed_energy + 1.20 * self.contradiction_ema
        )

    def axis_vector(self, ordered_names: Sequence[str]) -> np.ndarray:
        return np.array(
            [self.axis_state.get(name, 0.0) for name in ordered_names], dtype=np.float32
        )


@dataclass
class RepairCandidate:
    mode: str
    new_vector: np.ndarray
    new_axes: Dict[str, float]
    new_active: float
    new_repressed: float
    coherence_gain: float
    identity_drift: float
    pressure_relief: float
    reality_fit: float
    dream_support: float
    utility: float
    explanation: str


@dataclass
class LatentFragment:
    plot_id: str
    ts: float
    activation: float
    unresolved: float
    embedding: np.ndarray
    axis_evidence: Dict[str, float]
    source: str
    tags: Tuple[str, ...]


# ============================================================================
# Subconscious / dreams
# ============================================================================


class SubconsciousField:
    def __init__(self, reservoir: int = 256, seed: int = 0):
        self.reservoir = reservoir
        self.rng = np.random.default_rng(seed)
        self.fragments: List[LatentFragment] = []

    def add(self, fragment: LatentFragment) -> None:
        if len(self.fragments) < self.reservoir:
            self.fragments.append(fragment)
            return
        idx = int(self.rng.integers(0, len(self.fragments) + 1))
        if idx < self.reservoir:
            self.fragments[idx] = fragment

    def sample(self, n: int = 2) -> List[LatentFragment]:
        if not self.fragments:
            return []
        frags = self.fragments
        weights = []
        for frag in frags:
            age = max(1.0, now_ts() - frag.ts)
            rec = 1.0 / math.log1p(age)
            w = 0.55 * frag.unresolved + 0.25 * frag.activation + 0.20 * rec
            weights.append(max(1e-6, w))
        chosen: List[LatentFragment] = []
        pool = list(range(len(frags)))
        cur_weights = weights[:]
        n = min(n, len(frags))
        for _ in range(n):
            idx = weighted_choice(pool, [cur_weights[i] for i in pool], self.rng)
            chosen.append(frags[idx])
            pool.remove(idx)
        return chosen

    def operator(self, frags: Sequence[LatentFragment], state: IdentityState) -> str:
        unresolved = np.mean([f.unresolved for f in frags]) if frags else 0.0
        if unresolved > 0.72:
            return weighted_choice(
                ["fear_rehearsal", "counterfactual", "integration"],
                [0.42, 0.30, 0.28],
                self.rng,
            )
        if state.plasticity() > 0.58:
            return weighted_choice(
                ["integration", "wish_rehearsal", "counterfactual"],
                [0.42, 0.34, 0.24],
                self.rng,
            )
        return weighted_choice(
            ["counterfactual", "wish_rehearsal", "integration"],
            [0.35, 0.35, 0.30],
            self.rng,
        )

    def synthesize(
        self,
        frags: Sequence[LatentFragment],
        state: IdentityState,
        schema: PsychologicalSchema,
        narrator: NarrativeProvider,
    ) -> Tuple[str, EventFrame, float]:
        if not frags:
            frame = EventFrame(tags=("empty_dream",), arousal=0.05, self_relevance=0.8)
            return "梦里只有没有说完的话。", frame, 0.0

        op = self.operator(frags, state)
        text = narrator.compose_dream(op, frags, state, schema)

        # Blend fragment evidence into dream evidence.
        axis_names = schema.ordered_axis_names()
        blended: Dict[str, float] = {name: 0.0 for name in axis_names}
        for frag in frags:
            for name, val in frag.axis_evidence.items():
                blended[name] = blended.get(name, 0.0) + val / max(len(frags), 1)

        if op == "counterfactual":
            # push slightly toward agency / coherence if available
            if "agency" in blended:
                blended["agency"] = clamp(blended["agency"] + 0.18)
            if "coherence" in blended:
                blended["coherence"] = clamp(blended["coherence"] + 0.12)
        elif op == "fear_rehearsal":
            if "vigilance" in blended:
                blended["vigilance"] = clamp(blended["vigilance"] + 0.18)
            if "regulation" in blended:
                blended["regulation"] = clamp(blended["regulation"] - 0.08)
        elif op == "wish_rehearsal":
            if "affiliation" in blended:
                blended["affiliation"] = clamp(blended["affiliation"] + 0.12)
            if "regulation" in blended:
                blended["regulation"] = clamp(blended["regulation"] + 0.10)
        else:
            if "coherence" in blended:
                blended["coherence"] = clamp(blended["coherence"] + 0.18)
            if "regulation" in blended:
                blended["regulation"] = clamp(blended["regulation"] + 0.10)

        care = max(
            0.0,
            np.mean(
                [0.25 + 0.25 * max(0.0, f.axis_evidence.get("affiliation", 0.0)) for f in frags]
            ),
        )
        threat = max(
            0.0,
            np.mean([0.25 + 0.25 * max(0.0, f.axis_evidence.get("vigilance", 0.0)) for f in frags]),
        )
        valence = clamp(
            np.mean([sum(f.axis_evidence.values()) / max(len(f.axis_evidence), 1) for f in frags])
        )
        arousal = clamp01(0.30 + 0.25 * np.mean([f.unresolved for f in frags]))

        tags: List[str] = ["dream", op]
        for f in frags:
            for t in f.tags:
                if t not in tags:
                    tags.append(t)

        dream_frame = EventFrame(
            axis_evidence=blended,
            valence=valence,
            arousal=arousal,
            care=care,
            threat=threat,
            control=0.0,
            abandonment=0.0,
            agency_signal=max(0.0, blended.get("agency", 0.0)),
            shame=max(0.0, -valence) * 0.15,
            novelty=0.40,
            self_relevance=0.85,
            tags=tuple(tags[:12]),
        )
        resonance = np.mean(
            [
                cosine_sim(state.self_vector, frag.embedding) * (0.5 + frag.unresolved)
                for frag in frags
            ]
        )
        return text, dream_frame, float(resonance)


# ============================================================================
# Schema evolution
# ============================================================================


class SchemaEvolver:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.unresolved_tag_counts: Dict[str, float] = {}

    def observe(self, plot: Plot, schema: PsychologicalSchema) -> None:
        # High-tension unresolved tags accumulate as candidates for new persona axes.
        tension = plot.tension + 0.5 * plot.contradiction
        if plot.source != "wake" or tension < 0.75:
            return
        existing_names = set(schema.all_axes().keys())
        for tag in plot.frame.tags:
            if ":" in tag or tag.startswith("echo:"):
                continue
            if tag in existing_names:
                continue
            self.unresolved_tag_counts[tag] = (
                self.unresolved_tag_counts.get(tag, 0.0) + 0.25 + 0.25 * tension
            )

    def maybe_expand(
        self, schema: PsychologicalSchema, embedder: HashEmbedding
    ) -> Optional[AxisSpec]:
        if not self.unresolved_tag_counts:
            return None
        tag, score = max(self.unresolved_tag_counts.items(), key=lambda kv: kv[1])
        if score < 1.4:
            return None
        if tag in schema.all_axes():
            self.unresolved_tag_counts.pop(tag, None)
            return None
        neg = ANTONYM_HINTS.get(tag, f"not_{tag}")
        axis = AxisSpec(
            name=re.sub(r"\W+", "_", tag.lower())[:24] or f"axis_{len(schema.persona_axes) + 1}",
            positive_pole=tag,
            negative_pole=neg,
            description=f"Discovered persona dimension from repeated unresolved motif: {tag}",
            level="persona",
        )
        schema.add_persona_axis(axis, embedder)
        self.unresolved_tag_counts.pop(tag, None)
        return axis


# ============================================================================
# Main engine configuration
# ============================================================================


@dataclass
class GenerativeSoulConfig:
    dim: int = 384
    metric_rank: int = 32
    kde_reservoir: int = 1024
    story_alpha: float = 1.2
    theme_alpha: float = 0.9
    subconscious_reservoir: int = 256
    encode_min_events_before_gating: int = 6
    mode_refractory_steps: int = 4
    mode_new_threshold: float = 0.52
    max_recent_texts: int = 12
    profile_text: str = ""
    persona_axes: Optional[List[Dict[str, Any]]] = None


# ============================================================================
# Main engine
# ============================================================================


class AuroraGenerativeSoulMemory:
    """
    Public API:
        ingest(text, actors=None, source="wake", confidence=1.0, evidence_weight=1.0) -> Plot
        evolve(dreams=2) -> List[Plot]
        query(text, k=6) -> RetrievalTrace
        feedback_retrieval(query_text, chosen_id, success) -> None
        snapshot_identity() -> Dict[str, Any]
        narrative_summary() -> str

    Architectural stance:
    - keep the latent memory mathematics
    - replace persona-specific hardcoding with dynamic schema + dynamic modes
    - keep load-bearing invariants
    """

    def __init__(
        self,
        cfg: GenerativeSoulConfig = GenerativeSoulConfig(),
        seed: int = 0,
        meaning_provider: Optional[MeaningProvider] = None,
        narrator: Optional[NarrativeProvider] = None,
    ):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        self.embedder = HashEmbedding(dim=cfg.dim, seed=seed)
        self.schema = PsychologicalSchema.from_profile(
            embedder=self.embedder,
            profile_text=cfg.profile_text,
            persona_axes=cfg.persona_axes,
        )
        self.meaning_provider = meaning_provider or HeuristicMeaningProvider()
        self.narrator = narrator or CombinatorialNarrator()

        self.kde = OnlineKDE(dim=cfg.dim, reservoir=cfg.kde_reservoir, seed=seed)
        self.metric = LowRankMetric(dim=cfg.dim, rank=cfg.metric_rank, seed=seed)
        self.gate = ThompsonEncodeGate(seed=seed)

        self.graph = MemoryGraph()
        self.vindex = VectorIndex(dim=cfg.dim)
        self.retriever = FieldRetriever(metric=self.metric, vindex=self.vindex, graph=self.graph)
        self.story_model = StoryModel(metric=self.metric)
        self.theme_model = ThemeModel(metric=self.metric)
        self.crp_story = CRPAssigner(alpha=cfg.story_alpha, seed=seed)
        self.crp_theme = CRPAssigner(alpha=cfg.theme_alpha, seed=seed + 1)
        self.subconscious = SubconsciousField(reservoir=cfg.subconscious_reservoir, seed=seed)
        self.schema_evolver = SchemaEvolver(seed=seed + 2)

        self.plots: Dict[str, Plot] = {}
        self.stories: Dict[str, StoryArc] = {}
        self.themes: Dict[str, Theme] = {}
        self.modes: Dict[str, IdentityMode] = {}
        self.recent_texts: List[str] = []
        self.step = 0

        self.wake_axis_stats: Dict[str, Dict[str, float]] = {}
        self.axis_names_cache: List[str] = self.schema.ordered_axis_names()
        for name in self.axis_names_cache:
            self.wake_axis_stats[name] = {"pos": 1.0, "neg": 1.0}

        self.identity = self._make_initial_identity()
        self._init_origin_mode()

    # ----------------------------------------------------------------------
    # initialization
    # ----------------------------------------------------------------------

    def _make_initial_identity(self) -> IdentityState:
        axis_state = {name: 0.0 for name in self.schema.ordered_axis_names()}

        # A small bias can be derived from profile axis poles.
        for name, axis in self.schema.persona_axes.items():
            axis_state[name] = 0.20

        # Homeostatic defaults are neutral but lightly coherent.
        axis_state["coherence"] = 0.15
        axis_state["regulation"] = 0.10
        self_vec = self._compose_self_vector(axis_state)
        state = IdentityState(
            self_vector=self_vec,
            axis_state=axis_state,
            intuition_axes={name: 0.0 for name in axis_state},
            summary="初始自我尚未充分成形，但已经开始维持一条连续叙事。",
            current_mode_label="origin",
        )
        return state

    def _compose_self_vector(self, axis_state: Dict[str, float]) -> np.ndarray:
        acc = np.zeros(self.cfg.dim, dtype=np.float32)
        for name, value in axis_state.items():
            axis = self.schema.all_axes().get(name)
            if not axis or axis.direction is None:
                continue
            acc += float(value) * axis.direction
        if np.linalg.norm(acc) < 1e-6:
            acc += self.embedder.encode(self.cfg.profile_text or "origin self")
        return l2_normalize(acc)

    def _init_origin_mode(self) -> None:
        mode_id = stable_uuid("mode")
        axis_proto = {k: float(v) for k, v in self.identity.axis_state.items()}
        label = self.narrator.label_mode(axis_proto, self.schema, support=1)
        mode = IdentityMode(
            id=mode_id,
            label=label,
            prototype=self.identity.self_vector.copy(),
            axis_prototype=axis_proto,
            support=1,
            barrier=0.55,
            hysteresis=0.08,
        )
        self.modes[mode_id] = mode
        self.identity.current_mode_id = mode_id
        self.identity.current_mode_label = label
        self.identity.summary = self.narrator.compose_summary(
            self.identity, self.schema, self.recent_texts
        )
        self.identity.narrative_log.append(f"初始化模式：{label}")

    # ----------------------------------------------------------------------
    # helpers
    # ----------------------------------------------------------------------

    def _axis_names(self) -> List[str]:
        # refresh if schema expanded
        names = self.schema.ordered_axis_names()
        for name in names:
            if name not in self.identity.axis_state:
                self.identity.axis_state[name] = 0.0
                self.identity.intuition_axes[name] = 0.0
                self.wake_axis_stats[name] = {"pos": 1.0, "neg": 1.0}
        self.axis_names_cache = names
        return names

    def _plot_vector_for_index(self, plot: Plot) -> np.ndarray:
        # Blend event embedding with self-implication vector for retrieval that respects identity.
        impl = np.zeros(self.cfg.dim, dtype=np.float32)
        for name, val in plot.frame.axis_evidence.items():
            axis = self.schema.all_axes().get(name)
            if axis is not None and axis.direction is not None:
                impl += float(val) * axis.direction
        if np.linalg.norm(impl) < 1e-6:
            return plot.embedding
        return l2_normalize(0.72 * plot.embedding + 0.28 * impl)

    def _story_vector_for_index(self, story: StoryArc) -> np.ndarray:
        if story.centroid is None:
            return np.zeros(self.cfg.dim, dtype=np.float32)
        return story.centroid

    def _theme_vector_for_index(self, theme: Theme) -> np.ndarray:
        if theme.prototype is None:
            return np.zeros(self.cfg.dim, dtype=np.float32)
        return theme.prototype

    # ----------------------------------------------------------------------
    # dissonance / repair
    # ----------------------------------------------------------------------

    def _compute_dissonance(self, frame: EventFrame, embedding: np.ndarray) -> DissonanceReport:
        axis_conflicts: Dict[str, float] = {}
        axis_alignments: Dict[str, float] = {}
        for name in self._axis_names():
            current = self.identity.axis_state.get(name, 0.0)
            ev = frame.axis_evidence.get(name, 0.0)
            axis_conflicts[name] = max(0.0, -current * ev)
            axis_alignments[name] = max(0.0, current * ev)

        semantic_conflict = (
            max(0.0, -cosine_sim(self.identity.self_vector, embedding)) * frame.self_relevance
        )
        affective_load = (
            0.30 * frame.arousal
            + 0.20 * max(0.0, -frame.valence)
            + 0.18 * frame.threat
            + 0.12 * frame.shame
            + 0.10 * frame.control
            + 0.10 * frame.abandonment
        )
        narrative_incongruity = mean_abs(axis_conflicts.values()) * (
            0.55 + 0.45 * frame.self_relevance
        )
        total = (
            0.36 * mean_abs(axis_conflicts.values())
            + 0.20 * semantic_conflict
            + 0.28 * affective_load
            + 0.16 * narrative_incongruity
        )
        return DissonanceReport(
            axis_conflicts=axis_conflicts,
            axis_alignments=axis_alignments,
            semantic_conflict=semantic_conflict,
            affective_load=affective_load,
            narrative_incongruity=narrative_incongruity,
            total=float(total),
        )

    def _dynamic_repair_threshold(self) -> float:
        # lower threshold when regulation/coherence are low or pressure already high.
        # This engine aims to produce phase-like nonlinearity without requiring absurdly
        # large scalar conflicts before narrative reconstruction can start.
        plasticity = self.identity.plasticity()
        rigidity = self.identity.rigidity()
        base = (
            0.42 + 0.14 * rigidity - 0.10 * plasticity + 0.03 * min(self.identity.repair_count, 6)
        )
        return max(0.26, base)

    def _top_conflict_axes(self, dissonance: DissonanceReport, topn: int = 3) -> List[str]:
        items = sorted(dissonance.axis_conflicts.items(), key=lambda kv: kv[1], reverse=True)
        return [k for k, v in items[:topn] if v > 0.03]

    def _reality_fit_for_axis(self, axis_name: str, sign: float) -> float:
        stats = self.wake_axis_stats.get(axis_name, {"pos": 1.0, "neg": 1.0})
        pos = stats["pos"]
        neg = stats["neg"]
        if sign >= 0:
            return pos / (pos + neg + 1e-12)
        return neg / (pos + neg + 1e-12)

    def _candidate_repairs(self, plot: Plot, dissonance: DissonanceReport) -> List[RepairCandidate]:
        state = self.identity
        salient_axes = self._top_conflict_axes(dissonance, topn=4)
        if not salient_axes:
            fallback_scores: Dict[str, float] = {}
            for name in self._axis_names():
                base = abs(plot.frame.axis_evidence.get(name, 0.0))
                if name == "vigilance":
                    base += 0.22 * plot.frame.threat
                elif name == "agency":
                    base += 0.18 * (plot.frame.control + plot.frame.agency_signal)
                elif name == "affiliation":
                    base += 0.16 * (plot.frame.care + plot.frame.abandonment)
                elif name == "coherence":
                    base += 0.18 * plot.frame.arousal
                elif name == "regulation":
                    base += 0.16 * (plot.frame.arousal + plot.frame.shame)
                fallback_scores[name] = base
            salient_axes = [
                k
                for k, v in sorted(fallback_scores.items(), key=lambda kv: kv[1], reverse=True)[:4]
                if v > 0.08
            ]
            if not salient_axes:
                return []

        axis_order = self._axis_names()
        dream_support = 0.0
        for name in axis_order:
            intuition = state.intuition_axes.get(name, 0.0)
            dream_support += intuition * plot.frame.axis_evidence.get(name, 0.0)
        dream_support /= max(len(axis_order), 1)

        repeated_conflict = (
            np.mean(
                [
                    self._reality_fit_for_axis(name, plot.frame.axis_evidence.get(name, 0.0))
                    for name in salient_axes
                ]
            )
            if salient_axes
            else 0.0
        )

        def base_copy() -> Dict[str, float]:
            return {k: float(v) for k, v in state.axis_state.items()}

        def shift_axes(mode: str) -> Tuple[Dict[str, float], np.ndarray]:
            axes = base_copy()

            # generic axis moves
            for name in salient_axes:
                ev = plot.frame.axis_evidence.get(name, 0.0)
                cur = axes.get(name, 0.0)
                if mode == "preserve":
                    if cur * ev < 0:
                        axes[name] = clamp(cur + 0.18 * np.sign(cur if abs(cur) > 0.02 else -ev))
                elif mode == "reframe":
                    axes[name] = clamp(cur + 0.10 * ev * (0.5 + state.plasticity()))
                elif mode == "revise":
                    axes[name] = clamp(cur + 0.22 * ev * (0.45 + repeated_conflict))
                elif mode == "differentiate":
                    axes[name] = clamp(cur + 0.10 * ev * 0.5)
                else:  # integrate
                    axes[name] = clamp(
                        cur + 0.16 * ev * (0.35 + repeated_conflict + max(0.0, dream_support))
                    )

            # universal regulatory moves
            rel_threat = plot.frame.threat + 0.7 * plot.frame.control + 0.6 * plot.frame.abandonment
            care = plot.frame.care
            agency = plot.frame.agency_signal

            if mode == "preserve":
                axes["vigilance"] = clamp(axes.get("vigilance", 0.0) + 0.18 * rel_threat)
                axes["exploration"] = clamp(axes.get("exploration", 0.0) - 0.12 * rel_threat)
                axes["regulation"] = clamp(axes.get("regulation", 0.0) - 0.04 * plot.frame.arousal)
            elif mode == "reframe":
                axes["coherence"] = clamp(axes.get("coherence", 0.0) + 0.10)
                axes["regulation"] = clamp(axes.get("regulation", 0.0) + 0.06)
            elif mode == "revise":
                axes["coherence"] = clamp(axes.get("coherence", 0.0) + 0.14)
                axes["exploration"] = clamp(axes.get("exploration", 0.0) + 0.08 * repeated_conflict)
            elif mode == "differentiate":
                axes["agency"] = clamp(axes.get("agency", 0.0) + 0.22 * max(agency, rel_threat))
                axes["affiliation"] = clamp(axes.get("affiliation", 0.0) - 0.10 * rel_threat)
                axes["vigilance"] = clamp(axes.get("vigilance", 0.0) + 0.12 * rel_threat)
                axes["coherence"] = clamp(axes.get("coherence", 0.0) + 0.06)
            else:  # integrate
                axes["coherence"] = clamp(axes.get("coherence", 0.0) + 0.16)
                axes["regulation"] = clamp(axes.get("regulation", 0.0) + 0.12)
                axes["exploration"] = clamp(
                    axes.get("exploration", 0.0) + 0.08 * max(care, repeated_conflict)
                )
                axes["agency"] = clamp(axes.get("agency", 0.0) + 0.06 * agency)
                axes["affiliation"] = clamp(
                    axes.get("affiliation", 0.0) + 0.08 * care - 0.04 * rel_threat
                )

            new_vec = self._compose_self_vector(axes)
            return axes, new_vec

        def score(
            mode: str,
            axes: Dict[str, float],
            new_vec: np.ndarray,
            active_after: float,
            repressed_after: float,
        ) -> RepairCandidate:
            drift = 1.0 - cosine_sim(state.self_vector, new_vec)
            coherence_gain = max(
                0.0, axes.get("coherence", 0.0) - state.axis_state.get("coherence", 0.0)
            )
            current_pressure = state.narrative_pressure()
            new_contradiction_ema = moving_average(state.contradiction_ema, dissonance.total, 0.20)
            projected_pressure = (
                active_after + 0.85 * repressed_after + 1.20 * new_contradiction_ema
            )
            pressure_relief = max(0.0, current_pressure - projected_pressure)

            reality = (
                np.mean(
                    [self._reality_fit_for_axis(name, axes.get(name, 0.0)) for name in salient_axes]
                )
                if salient_axes
                else 0.5
            )

            # Penalize unconstrained drift unless repeated evidence supports it.
            continuity_penalty = drift * (0.85 - 0.45 * repeated_conflict)

            utility = (
                0.36 * pressure_relief
                + 0.22 * coherence_gain
                + 0.18 * reality
                + 0.10 * max(0.0, dream_support)
                - 0.16 * continuity_penalty
            )

            if mode == "preserve":
                utility += 0.10 * state.rigidity() + 0.10 * max(0.0, plot.frame.threat)
            elif mode == "reframe":
                utility += 0.08 * state.plasticity()
            elif mode == "revise":
                utility += 0.16 * repeated_conflict + 0.08 * plot.evidence_weight
            elif mode == "differentiate":
                utility += 0.14 * (
                    plot.frame.threat + plot.frame.control + plot.frame.agency_signal
                )
            else:  # integrate
                utility += (
                    0.12 * state.plasticity()
                    + 0.12 * max(0.0, dream_support)
                    + 0.06 * plot.frame.care
                )

            temp_state = copy.deepcopy(state)
            temp_state.axis_state = axes
            temp_state.self_vector = new_vec
            explanation = self.narrator.compose_repair(
                mode=mode,
                state_before=state,
                state_after=temp_state,
                dissonance=dissonance,
                salient_axes=salient_axes,
                plot=plot,
                schema=self.schema,
            )

            return RepairCandidate(
                mode=mode,
                new_vector=new_vec,
                new_axes=axes,
                new_active=max(0.0, active_after),
                new_repressed=max(0.0, repressed_after),
                coherence_gain=coherence_gain,
                identity_drift=drift,
                pressure_relief=pressure_relief,
                reality_fit=float(reality),
                dream_support=float(dream_support),
                utility=float(utility),
                explanation=explanation,
            )

        cands: List[RepairCandidate] = []

        # preserve
        axes, new_vec = shift_axes("preserve")
        cands.append(
            score(
                "preserve",
                axes,
                new_vec,
                active_after=state.active_energy - 0.28 * dissonance.total,
                repressed_after=state.repressed_energy + 0.52 * dissonance.total,
            )
        )

        # reframe
        axes, new_vec = shift_axes("reframe")
        cands.append(
            score(
                "reframe",
                axes,
                new_vec,
                active_after=state.active_energy - 0.48 * dissonance.total,
                repressed_after=state.repressed_energy + 0.16 * dissonance.total,
            )
        )

        # revise
        axes, new_vec = shift_axes("revise")
        cands.append(
            score(
                "revise",
                axes,
                new_vec,
                active_after=state.active_energy - 0.72 * dissonance.total,
                repressed_after=state.repressed_energy + 0.06 * dissonance.total,
            )
        )

        # differentiate
        axes, new_vec = shift_axes("differentiate")
        cands.append(
            score(
                "differentiate",
                axes,
                new_vec,
                active_after=state.active_energy - 0.60 * dissonance.total,
                repressed_after=state.repressed_energy + 0.12 * dissonance.total,
            )
        )

        # integrate
        axes, new_vec = shift_axes("integrate")
        cands.append(
            score(
                "integrate",
                axes,
                new_vec,
                active_after=state.active_energy - 0.82 * dissonance.total,
                repressed_after=max(0.0, state.repressed_energy - 0.10 * dissonance.total),
            )
        )

        return cands

    def _apply_repair(self, cand: RepairCandidate, plot: Plot) -> None:
        self.identity.self_vector = cand.new_vector
        self.identity.axis_state = copy.deepcopy(cand.new_axes)
        self.identity.active_energy = cand.new_active
        self.identity.repressed_energy = cand.new_repressed
        self.identity.repair_count += 1
        self.identity.narrative_log.append(cand.explanation)

        repair_plot = self.ingest(
            text=f"[repair:{cand.mode}] {cand.explanation}",
            actors=("self",),
            source="repair",
            confidence=0.72,
            evidence_weight=0.35,
        )
        # link repair to source plot only if both are stored
        if repair_plot.id in self.graph.g and plot.id in self.graph.g:
            self.graph.ensure_edge(repair_plot.id, plot.id, "rationalizes")

    def _maybe_reconstruct(self, plot: Plot, dissonance: DissonanceReport) -> None:
        pressure = self.identity.narrative_pressure()
        threshold = self._dynamic_repair_threshold()
        activation = pressure + 0.65 * plot.tension + 0.75 * dissonance.total
        p_repair = sigmoid(activation - threshold)

        must = activation > threshold
        if not must and self.rng.random() >= p_repair:
            self.identity.repressed_energy += 0.22 * dissonance.total
            return

        cands = self._candidate_repairs(plot, dissonance)
        if not cands:
            self.identity.repressed_energy += 0.15 * dissonance.total
            return
        probs = softmax(
            [c.utility for c in cands],
            temperature=max(0.35, 0.95 - 0.40 * self.identity.plasticity()),
        )
        idx = int(self.rng.choice(np.arange(len(cands)), p=np.array(probs, dtype=np.float64)))
        self._apply_repair(cands[idx], plot)

    # ----------------------------------------------------------------------
    # dynamic mode discovery / transitions
    # ----------------------------------------------------------------------

    def _current_axis_signature(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.identity.axis_state.items()}

    def _maybe_mode_transition(self, cause_plot: Plot) -> Optional[IdentityMode]:
        if self.step - self.identity.last_mode_step < self.cfg.mode_refractory_steps:
            return None

        pressure = self.identity.narrative_pressure()
        cur_mode = (
            self.modes.get(self.identity.current_mode_id) if self.identity.current_mode_id else None
        )
        current_score = (
            cur_mode.score(self.identity.self_vector, self.identity.axis_state) if cur_mode else 0.0
        )

        best_mode: Optional[IdentityMode] = cur_mode
        best_score = current_score

        for mode in self.modes.values():
            sc = mode.score(self.identity.self_vector, self.identity.axis_state)
            if sc > best_score:
                best_score = sc
                best_mode = mode

        # New mode emergence condition
        drift_from_current = 1.0 - (
            cur_mode.score(self.identity.self_vector, self.identity.axis_state) if cur_mode else 0.0
        )
        create_new = (
            pressure > 0.95
            and drift_from_current > self.cfg.mode_new_threshold
            and (best_mode is cur_mode or best_score < 0.76)
        )

        if create_new:
            mode_id = stable_uuid("mode")
            axis_sig = self._current_axis_signature()
            label = self.narrator.label_mode(axis_sig, self.schema, support=1)
            new_mode = IdentityMode(
                id=mode_id,
                label=label,
                prototype=self.identity.self_vector.copy(),
                axis_prototype=axis_sig,
                support=1,
                barrier=0.52 + 0.10 * self.identity.rigidity(),
                hysteresis=0.06 + 0.06 * self.identity.rigidity(),
            )
            self.modes[mode_id] = new_mode
            best_mode = new_mode
            best_score = new_mode.score(self.identity.self_vector, self.identity.axis_state)

        if best_mode is None:
            return None
        if cur_mode is not None and best_mode.id == cur_mode.id:
            # reinforce prototype
            cur_mode.prototype = l2_normalize(
                0.92 * cur_mode.prototype + 0.08 * self.identity.self_vector
            )
            for k, v in self.identity.axis_state.items():
                cur_mode.axis_prototype[k] = moving_average(
                    cur_mode.axis_prototype.get(k, 0.0), v, 0.08
                )
            cur_mode.support += 1
            cur_mode.updated_ts = now_ts()
            return None

        barrier = best_mode.barrier
        hysteresis = best_mode.hysteresis
        margin = best_score - current_score
        threshold = barrier + hysteresis * (1.0 + 0.35 * self.identity.mode_change_count)
        if margin <= threshold:
            return None

        # transition
        prev_label = self.identity.current_mode_label
        self.identity.current_mode_id = best_mode.id
        self.identity.current_mode_label = best_mode.label
        self.identity.mode_change_count += 1
        self.identity.last_mode_step = self.step
        self.identity.last_mode_change_ts = now_ts()
        self.identity.narrative_log.append(f"模式跃迁：{prev_label} -> {best_mode.label}")

        mode_plot = self.ingest(
            text=f"[mode] 她从「{prev_label}」进入「{best_mode.label}」。",
            actors=("self",),
            source="mode",
            confidence=0.76,
            evidence_weight=0.40,
        )
        if mode_plot.id in self.graph.g and cause_plot.id in self.graph.g:
            self.graph.ensure_edge(mode_plot.id, cause_plot.id, "triggered_by")
        return best_mode

    # ----------------------------------------------------------------------
    # plotting / storage
    # ----------------------------------------------------------------------

    def _assign_story(self, plot: Plot) -> StoryArc:
        logps: Dict[str, float] = {}
        for sid, story in self.stories.items():
            log_prior = math.log(len(story.plot_ids) + 1.0)
            log_lik = self.story_model.loglik(plot, story, self.plots)
            logps[sid] = log_prior + log_lik
        sid, _ = self.crp_story.sample(logps)
        if sid is None:
            sid = stable_uuid("story")
            story = StoryArc(id=sid, created_ts=plot.ts, updated_ts=plot.ts)
            self.stories[sid] = story
            self.graph.add_node(sid, "story", story)
            self.vindex.add(sid, plot.embedding, "story")
        story = self.stories[sid]
        self._update_story(story, plot)
        plot.story_id = sid
        self.graph.ensure_edge(plot.id, sid, "belongs_to")
        self.graph.ensure_edge(sid, plot.id, "contains")
        return story

    def _update_story(self, story: StoryArc, plot: Plot) -> None:
        if story.centroid is None:
            story.centroid = plot.embedding.copy()
        else:
            d2 = self.metric.d2(plot.embedding, story.centroid)
            story._update_stat("dist", d2)
            story.centroid = l2_normalize(0.90 * story.centroid + 0.10 * plot.embedding)
        if story.plot_ids:
            gap = max(0.0, plot.ts - story.updated_ts)
            story._update_stat("gap", gap)
        story.plot_ids.append(plot.id)
        story.updated_ts = plot.ts
        for a in plot.actors:
            story.actor_counts[a] = story.actor_counts.get(a, 0) + 1
        for tg in plot.frame.tags[:8]:
            story.tag_counts[tg] = story.tag_counts.get(tg, 0) + 1
        story.source_counts[plot.source] = story.source_counts.get(plot.source, 0) + 1
        story.tension_curve.append(float(plot.tension))
        if story.id in self.vindex.ids:
            idx = self.vindex.ids.index(story.id)
            self.vindex.vecs[idx] = self._story_vector_for_index(story)

    def _assign_theme(self, story: StoryArc) -> Optional[Theme]:
        if story.centroid is None:
            return None
        logps: Dict[str, float] = {}
        for tid, theme in self.themes.items():
            log_prior = math.log(len(theme.story_ids) + 1.0)
            log_lik = self.theme_model.loglik(story, theme)
            logps[tid] = log_prior + log_lik
        tid, _ = self.crp_theme.sample(logps)
        if tid is None:
            tid = stable_uuid("theme")
            label = self._label_theme(story)
            theme = Theme(
                id=tid,
                created_ts=now_ts(),
                updated_ts=now_ts(),
                story_ids=[],
                prototype=story.centroid.copy(),
                label=label,
                description=f"Emergent theme around: {label}",
            )
            self.themes[tid] = theme
            self.graph.add_node(tid, "theme", theme)
            self.vindex.add(tid, theme.prototype, "theme")
        theme = self.themes[tid]
        if story.id not in theme.story_ids:
            theme.story_ids.append(story.id)
        theme.updated_ts = now_ts()
        if theme.prototype is None:
            theme.prototype = story.centroid.copy()
        else:
            theme.prototype = l2_normalize(0.90 * theme.prototype + 0.10 * story.centroid)
        self.graph.ensure_edge(story.id, tid, "instantiates")
        self.graph.ensure_edge(tid, story.id, "grounds")
        if theme.id in self.vindex.ids:
            idx = self.vindex.ids.index(theme.id)
            self.vindex.vecs[idx] = self._theme_vector_for_index(theme)
        return theme

    def _label_theme(self, story: StoryArc) -> str:
        tags = sorted(story.tag_counts.items(), key=lambda kv: kv[1], reverse=True)
        if not tags:
            return "untitled_theme"
        return " / ".join([t for t, _ in tags[:3]])

    def _encode_features(self, plot: Plot) -> Dict[str, float]:
        return {
            "surprise": clamp01(plot.surprise / 3.0),
            "tension": clamp01(plot.tension / 2.0),
            "contradiction": clamp01(plot.contradiction / 1.5),
            "arousal": clamp01(plot.frame.arousal),
            "novelty": clamp01(plot.frame.novelty),
        }

    def _maybe_store_fragment(self, plot: Plot) -> None:
        unresolved = clamp01(0.55 * plot.contradiction + 0.45 * plot.tension)
        activation = clamp01(0.30 + 0.35 * plot.frame.arousal + 0.35 * plot.frame.self_relevance)
        frag = LatentFragment(
            plot_id=plot.id,
            ts=plot.ts,
            activation=activation,
            unresolved=unresolved,
            embedding=plot.embedding.copy(),
            axis_evidence={k: float(v) for k, v in plot.frame.axis_evidence.items()},
            source=plot.source,
            tags=tuple(plot.frame.tags[:10]),
        )
        self.subconscious.add(frag)

    # ----------------------------------------------------------------------
    # public ingest
    # ----------------------------------------------------------------------

    def ingest(
        self,
        text: str,
        actors: Optional[Sequence[str]] = None,
        source: Literal["wake", "dream", "repair", "mode"] = "wake",
        confidence: float = 1.0,
        evidence_weight: float = 1.0,
    ) -> Plot:
        self.step += 1
        actors = tuple(actors or ("user", "self"))
        emb = self.embedder.encode(text)
        frame = self.meaning_provider.extract(text, emb, self.schema, recent_tags=self.recent_texts)

        dissonance = self._compute_dissonance(frame, emb)
        surprise = self.kde.surprise(emb)

        redundancy = 0.0
        hits = self.vindex.search(emb, k=4, kind="plot")
        if hits:
            redundancy = max(sim for _, sim in hits)

        tension = (
            0.38 * dissonance.total
            + 0.25 * surprise
            + 0.18 * frame.arousal
            + 0.10 * frame.self_relevance
            + 0.09 * max(0.0, 1.0 - redundancy)
        ) * evidence_weight

        plot = Plot(
            id=stable_uuid("plot"),
            ts=now_ts(),
            text=text,
            actors=actors,
            embedding=emb,
            frame=frame,
            source=source,
            confidence=confidence,
            evidence_weight=evidence_weight,
            surprise=surprise,
            redundancy=redundancy,
            contradiction=dissonance.total,
            tension=tension,
        )

        features = self._encode_features(plot)
        should_encode = True
        if source in ("dream", "repair", "mode"):
            should_encode = True
        elif len(self.plots) >= self.cfg.encode_min_events_before_gating:
            should_encode = self.gate.should_encode(features, source)

        if should_encode:
            self._store_plot(plot)
            self.gate.update(source, features, success=True)
        else:
            self.gate.update(source, features, success=False)

        # Update state dynamics even for skipped storage if it is a wake event
        self.identity.active_energy += (
            0.38 * dissonance.total
            + 0.26 * frame.arousal
            + 0.16 * max(0.0, -frame.valence)
            + 0.10 * frame.threat
        )
        self.identity.contradiction_ema = moving_average(
            self.identity.contradiction_ema, dissonance.total, 0.24
        )
        if source == "wake":
            self._update_wake_axis_stats(frame)
            self._maybe_reconstruct(plot, dissonance)
            self._baseline_assimilation(frame)
            self.schema_evolver.observe(plot, self.schema)
            new_axis = self.schema_evolver.maybe_expand(self.schema, self.embedder)
            if new_axis is not None:
                self._axis_names()  # refresh caches
                self.identity.narrative_log.append(
                    f"新维度涌现：{new_axis.name} ({new_axis.positive_pole} ↔ {new_axis.negative_pole})"
                )
            self._promote_dream_intuitions()
            self._maybe_mode_transition(plot)

        self.identity.summary = self.narrator.compose_summary(
            self.identity, self.schema, self.recent_texts
        )
        self.recent_texts.append(text)
        if len(self.recent_texts) > self.cfg.max_recent_texts:
            self.recent_texts.pop(0)
        return plot

    def _store_plot(self, plot: Plot) -> None:
        self.plots[plot.id] = plot
        self.graph.add_node(plot.id, "plot", plot)
        self.vindex.add(plot.id, self._plot_vector_for_index(plot), "plot")
        self.kde.add(plot.embedding)

        story = self._assign_story(plot)
        theme = self._assign_theme(story)
        if theme is not None:
            plot.theme_id = theme.id
            self.graph.ensure_edge(plot.id, theme.id, "suggests_theme")
            self.graph.ensure_edge(theme.id, plot.id, "evidenced_by")

        self._maybe_store_fragment(plot)

    def _baseline_assimilation(self, frame: EventFrame) -> None:
        """
        Wake experience produces tiny continuous drift even without explicit repair.
        This keeps the system from becoming unrealistically inert while preserving
        phase-transition style updates for larger restructurings.
        """
        changed = False
        for name in self._axis_names():
            axis = self.schema.all_axes().get(name)
            ev = frame.axis_evidence.get(name, 0.0)
            if abs(ev) < 0.06:
                continue
            rate = 0.015 if (axis and axis.level == "persona") else 0.025
            rate *= 0.55 + 0.45 * frame.self_relevance
            self.identity.axis_state[name] = moving_average(
                self.identity.axis_state.get(name, 0.0),
                ev,
                rate,
            )
            changed = True

        # affective regulation channels respond weakly to raw signals
        self.identity.axis_state["regulation"] = clamp(
            self.identity.axis_state.get("regulation", 0.0)
            + 0.03 * frame.care
            - 0.04 * frame.arousal
            - 0.03 * frame.threat
        )
        self.identity.axis_state["vigilance"] = clamp(
            self.identity.axis_state.get("vigilance", 0.0) + 0.03 * frame.threat - 0.02 * frame.care
        )
        self.identity.axis_state["coherence"] = clamp(
            self.identity.axis_state.get("coherence", 0.0) - 0.02 * frame.shame + 0.01 * frame.care
        )
        changed = True
        if changed:
            self.identity.self_vector = self._compose_self_vector(self.identity.axis_state)

    def _update_wake_axis_stats(self, frame: EventFrame) -> None:
        for name in self._axis_names():
            val = frame.axis_evidence.get(name, 0.0)
            if val >= 0:
                self.wake_axis_stats[name]["pos"] += abs(val) + 1e-3
            else:
                self.wake_axis_stats[name]["neg"] += abs(val) + 1e-3

    # ----------------------------------------------------------------------
    # dreams
    # ----------------------------------------------------------------------

    def _dream_update_intuition(self, frame: EventFrame, resonance: float) -> None:
        gain = 0.10 + 0.12 * sigmoid(resonance)
        for name in self._axis_names():
            ev = frame.axis_evidence.get(name, 0.0)
            self.identity.intuition_axes[name] = moving_average(
                self.identity.intuition_axes.get(name, 0.0),
                gain * ev,
                0.14,
            )

    def _promote_dream_intuitions(self) -> None:
        for name in self._axis_names():
            bias = self.identity.intuition_axes.get(name, 0.0)
            if abs(bias) < 1e-6:
                continue
            corroboration = self._reality_fit_for_axis(name, bias)
            strength = abs(bias) * corroboration
            p = sigmoid(3.0 * (strength - 0.35))
            if self.rng.random() < p:
                self.identity.axis_state[name] = clamp(
                    self.identity.axis_state.get(name, 0.0) + 0.05 * bias
                )
        self.identity.self_vector = self._compose_self_vector(self.identity.axis_state)

    def dream(self, n: int = 2) -> List[Plot]:
        dream_plots: List[Plot] = []
        for _ in range(n):
            frags = self.subconscious.sample(n=int(self.rng.integers(1, 4)))
            if not frags:
                continue
            text, frame, resonance = self.subconscious.synthesize(
                frags, self.identity, self.schema, self.narrator
            )
            plot = self.ingest(
                text=text,
                actors=("self",),
                source="dream",
                confidence=0.34,
                evidence_weight=0.20,
            )
            # Use synthesized frame for intuition, not the fallback extractor output of the text alone.
            self._dream_update_intuition(frame, resonance)
            self.identity.dream_count += 1
            dream_plots.append(plot)
        if dream_plots:
            self._promote_dream_intuitions()
            self.identity.summary = self.narrator.compose_summary(
                self.identity, self.schema, self.recent_texts
            )
        return dream_plots

    def evolve(self, dreams: int = 2) -> List[Plot]:
        return self.dream(n=dreams)

    # ----------------------------------------------------------------------
    # retrieval / feedback
    # ----------------------------------------------------------------------

    def query(self, text: str, k: int = 6) -> RetrievalTrace:
        q_emb = self.embedder.encode(text)
        # Blend query with self-vector so retrieval is identity-conditioned.
        q = l2_normalize(0.75 * q_emb + 0.25 * self.identity.self_vector)
        trace = self.retriever.query(text, q, k=k)
        for hid, _ in trace.ranked:
            if hid in self.plots:
                self.plots[hid].access_count += 1
                self.plots[hid].last_access_ts = now_ts()
            elif hid in self.stories:
                self.stories[hid].reference_count += 1
        return trace

    def feedback_retrieval(self, query_text: str, chosen_id: str, success: bool) -> None:
        if chosen_id not in self.graph.g:
            return

        q = self.embedder.encode(query_text)
        chosen_vec = self.retriever._node_vector(chosen_id)
        coarse = self.vindex.search(q, k=6)
        neg_id = None
        for hid, _ in coarse:
            if hid != chosen_id:
                neg_id = hid
                break
        if neg_id is not None:
            neg_vec = self.retriever._node_vector(neg_id)
            if success:
                self.metric.update_triplet(q, chosen_vec, neg_vec, lr=0.02)
            else:
                self.metric.update_triplet(q, neg_vec, chosen_vec, lr=0.02)

        kind = self.graph.kind(chosen_id)
        if kind == "theme":
            self.themes[chosen_id].update_evidence(success)

        # Update graph edges around the chosen node.
        if chosen_id in self.graph.g:
            for nbr in list(self.graph.g.predecessors(chosen_id)) + list(
                self.graph.g.successors(chosen_id)
            ):
                belief = (
                    self.graph.edge_belief(nbr, chosen_id)
                    if self.graph.g.has_edge(nbr, chosen_id)
                    else None
                )
                if belief is not None:
                    belief.update(success)
                belief2 = (
                    self.graph.edge_belief(chosen_id, nbr)
                    if self.graph.g.has_edge(chosen_id, nbr)
                    else None
                )
                if belief2 is not None:
                    belief2.update(success)

    # ----------------------------------------------------------------------
    # state inspection
    # ----------------------------------------------------------------------

    def snapshot_identity(self) -> Dict[str, Any]:
        ordered_axes = self._axis_names()
        axes = {
            name: round(float(self.identity.axis_state.get(name, 0.0)), 4) for name in ordered_axes
        }
        intuition = {
            name: round(float(self.identity.intuition_axes.get(name, 0.0)), 4)
            for name in ordered_axes
        }
        modes = {}
        for mid, mode in self.modes.items():
            modes[mid] = {
                "label": mode.label,
                "support": mode.support,
                "barrier": round(mode.barrier, 4),
                "hysteresis": round(mode.hysteresis, 4),
            }
        return {
            "summary": self.identity.summary,
            "current_mode": self.identity.current_mode_label,
            "narrative_pressure": round(self.identity.narrative_pressure(), 4),
            "active_energy": round(self.identity.active_energy, 4),
            "repressed_energy": round(self.identity.repressed_energy, 4),
            "contradiction_ema": round(self.identity.contradiction_ema, 4),
            "plasticity": round(self.identity.plasticity(), 4),
            "rigidity": round(self.identity.rigidity(), 4),
            "axis_state": axes,
            "intuition_axes": intuition,
            "modes": modes,
            "persona_axes": {
                k: {
                    "positive_pole": v.positive_pole,
                    "negative_pole": v.negative_pole,
                    "description": v.description,
                }
                for k, v in self.schema.persona_axes.items()
            },
            "counts": {
                "plots": len(self.plots),
                "stories": len(self.stories),
                "themes": len(self.themes),
                "dreams": self.identity.dream_count,
                "repairs": self.identity.repair_count,
                "mode_changes": self.identity.mode_change_count,
            },
        }

    def narrative_summary(self) -> str:
        return self.identity.summary


# ============================================================================
# Helper presentation
# ============================================================================


def _top_axes_description(
    axis_state: Dict[str, float], schema: PsychologicalSchema, topn: int = 4
) -> str:
    items = sorted(axis_state.items(), key=lambda kv: abs(kv[1]), reverse=True)
    phrases = []
    for name, val in items[:topn]:
        ax = schema.all_axes().get(name)
        if not ax:
            continue
        pole = ax.positive_pole if val >= 0 else ax.negative_pole
        phrases.append(f"{pole}({val:+.2f})")
    return "、".join(phrases) if phrases else "尚未成形"


def _axes_to_phrase(axis_names: Sequence[str], schema: PsychologicalSchema) -> str:
    names = []
    for name in axis_names:
        ax = schema.all_axes().get(name)
        if ax:
            names.append(f"{ax.positive_pole}/{ax.negative_pole}")
        else:
            names.append(name)
    return "、".join(names) if names else "尚未命名的裂缝"


# ============================================================================
# Demo
# ============================================================================


def _demo_relational_arc() -> None:
    print("=" * 88)
    print("DEMO 1 — relational hurt -> repair -> dreams -> emergent mode")
    cfg = GenerativeSoulConfig(
        profile_text="敏感、想靠近、又害怕失去；受伤后会变得警觉，但也渴望理解"
    )
    mem = AuroraGenerativeSoulMemory(cfg=cfg, seed=7)

    events = [
        "她认真表达了自己的在乎，但对方突然冷淡下来，让她觉得像被推开。",
        "后来有人耐心听她讲完，没有评判，只是安静地陪着她。",
        "又一次，她发现对方隐瞒了事实，这让她非常警觉。",
        "她试着说出自己的边界：我可以靠近，但不能再被控制。",
        "有人尊重了她的边界，也没有离开，这让她第一次没有立刻后退。",
    ]
    for ev in events:
        p = mem.ingest(ev, actors=("other", "self"), source="wake")
        print(
            f"ingest: {p.text[:34]} | tension={p.tension:.3f} contradiction={p.contradiction:.3f}"
        )

    dreams = mem.evolve(dreams=3)
    for d in dreams:
        print(f"dream: {d.text[:56]}")

    snap = mem.snapshot_identity()
    print("mode:", snap["current_mode"])
    print("summary:", snap["summary"])
    print("axes:", json.dumps(snap["axis_state"], ensure_ascii=False, indent=2))


def _demo_scientist_profile() -> None:
    print("=" * 88)
    print("DEMO 2 — scientist profile with dynamic persona axes")
    cfg = GenerativeSoulConfig(
        profile_text="严谨的科学家，好奇但怀疑；重视证据、可重复性和解释力",
    )
    mem = AuroraGenerativeSoulMemory(cfg=cfg, seed=19)

    events = [
        "她发现实验结果很漂亮，但重复一次后没有复现，这让她立刻怀疑最初的结论。",
        "同事提供了更多数据，她开始建立更严格的控制组。",
        "有人催她赶快发表，但她拒绝了，因为证据还不够稳。",
        "夜里她反复想，如果最初的假设是错的，真正稳定的机制会是什么。",
    ]
    for ev in events[:-1]:
        p = mem.ingest(ev, actors=("colleague", "self"), source="wake")
        print(f"ingest: {p.text[:36]} | story={p.story_id} theme={p.theme_id}")

    dreams = mem.evolve(dreams=2)
    for d in dreams:
        print("dream:", d.text[:60])

    trace = mem.query("她怎么看待证据与假设的关系？", k=5)
    print("query evidence:")
    for line in trace.evidence:
        print("  ", line)

    snap = mem.snapshot_identity()
    print("persona axes:", json.dumps(snap["persona_axes"], ensure_ascii=False, indent=2))
    print("summary:", snap["summary"])


if __name__ == "__main__":
    _demo_relational_arc()
    _demo_scientist_profile()
