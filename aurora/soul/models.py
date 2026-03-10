from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np

from aurora.utils.math_utils import cosine_sim, l2_normalize, sigmoid, softmax
from aurora.utils.time_utils import now_ts

if TYPE_CHECKING:
    from aurora.soul.query import QueryType, TimeRange


HOMEOSTATIC_AXES: Tuple[Tuple[str, str, str, str], ...] = (
    ("affiliation", "closeness", "distance", "Need for closeness vs distance"),
    ("agency", "agency", "yielding", "Action, boundaries, self-authorship"),
    ("exploration", "curiosity", "closure", "Novelty seeking vs guarded closure"),
    ("vigilance", "vigilance", "ease", "Threat scanning vs trustful ease"),
    ("coherence", "integration", "fragmentation", "Narrative integration vs fragmentation"),
    ("regulation", "regulation", "reactivity", "Affect regulation vs reactivity"),
)


def stable_hash(text: str) -> int:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def mean_abs(xs: Iterable[float]) -> float:
    values = [abs(float(x)) for x in xs]
    return float(sum(values) / max(len(values), 1))


def vec_to_state(vec: Optional[np.ndarray]) -> Optional[List[float]]:
    if vec is None:
        return None
    return [float(x) for x in vec.astype(np.float32).tolist()]


def vec_from_state(values: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    if values is None:
        return None
    return np.asarray(values, dtype=np.float32)


def tokenize_loose(text: str) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z_\-]{1,}|[\u4e00-\u9fff]{1,6}|\d+", text.lower())
    return [word for word in words if word.strip()]


ANTONYM_HINTS: Dict[str, str] = {
    "严谨": "随意",
    "精密": "粗糙",
    "scientific": "improvised",
    "scientist": "intuitive",
    "curious": "closed",
    "skeptical": "gullible",
    "warm": "cold",
    "playful": "rigid",
    "protective": "exposed",
    "creative": "formulaic",
    "resilient": "fragile",
    "precise": "careless",
    "honest": "deceptive",
    "independent": "dependent",
}


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
    support_count: int = 1
    last_merged_ts: Optional[float] = None
    aliases: Tuple[str, ...] = ()
    positive_anchor: Optional[np.ndarray] = None
    negative_anchor: Optional[np.ndarray] = None
    direction: Optional[np.ndarray] = None

    def compile(self, embedder: Any) -> None:
        pos_text = " ".join(
            [self.name, self.positive_pole, self.description, *self.positive_examples, *self.aliases]
        ).strip() or self.positive_pole
        neg_text = " ".join(
            [self.name, self.negative_pole, self.description, *self.negative_examples]
        ).strip() or self.negative_pole
        self.positive_anchor = l2_normalize(embedder.embed(pos_text))
        self.negative_anchor = l2_normalize(embedder.embed(neg_text))
        self.direction = l2_normalize(self.positive_anchor - self.negative_anchor)

    def score(self, text_embedding: np.ndarray, text: str = "") -> float:
        if self.direction is None or self.positive_anchor is None or self.negative_anchor is None:
            raise ValueError(f"Axis {self.name} not compiled")
        sim_pos = cosine_sim(text_embedding, self.positive_anchor)
        sim_neg = cosine_sim(text_embedding, self.negative_anchor)
        base = clamp(sim_pos - sim_neg)

        # Keep a light lexical bonus for explainability and fallback robustness.
        text_lower = text.lower()
        for word in [self.positive_pole, *self.positive_examples, *self.aliases]:
            if word and word.lower() in text_lower:
                base += 0.05
        for word in [self.negative_pole, *self.negative_examples]:
            if word and word.lower() in text_lower:
                base -= 0.05
        return clamp(base)

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "positive_pole": self.positive_pole,
            "negative_pole": self.negative_pole,
            "description": self.description,
            "level": self.level,
            "weight": float(self.weight),
            "positive_examples": list(self.positive_examples),
            "negative_examples": list(self.negative_examples),
            "support_count": int(self.support_count),
            "last_merged_ts": self.last_merged_ts,
            "aliases": list(self.aliases),
            "positive_anchor": vec_to_state(self.positive_anchor),
            "negative_anchor": vec_to_state(self.negative_anchor),
            "direction": vec_to_state(self.direction),
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "AxisSpec":
        return cls(
            name=str(data["name"]),
            positive_pole=str(data["positive_pole"]),
            negative_pole=str(data["negative_pole"]),
            description=str(data.get("description", "")),
            level=str(data.get("level", "persona")),
            weight=float(data.get("weight", 1.0)),
            positive_examples=tuple(str(item) for item in data.get("positive_examples", [])),
            negative_examples=tuple(str(item) for item in data.get("negative_examples", [])),
            support_count=int(data.get("support_count", 1)),
            last_merged_ts=data.get("last_merged_ts"),
            aliases=tuple(str(item) for item in data.get("aliases", [])),
            positive_anchor=vec_from_state(data.get("positive_anchor")),
            negative_anchor=vec_from_state(data.get("negative_anchor")),
            direction=vec_from_state(data.get("direction")),
        )


@dataclass
class PsychologicalSchema:
    homeostatic_axes: Dict[str, AxisSpec] = field(default_factory=dict)
    persona_axes: Dict[str, AxisSpec] = field(default_factory=dict)
    profile_text: str = ""
    axis_aliases: Dict[str, str] = field(default_factory=dict)
    merge_history: List[Dict[str, Any]] = field(default_factory=list)

    def all_axes(self) -> Dict[str, AxisSpec]:
        axes = dict(self.homeostatic_axes)
        axes.update(self.persona_axes)
        return axes

    def ordered_axis_names(self) -> List[str]:
        return list(self.homeostatic_axes.keys()) + list(self.persona_axes.keys())

    def canonical_axis_name(self, axis_name: str) -> str:
        seen: set[str] = set()
        current = axis_name
        while current in self.axis_aliases and current not in seen:
            seen.add(current)
            current = self.axis_aliases[current]
        return current

    def compile(self, axis_embedder: Any) -> None:
        for axis in self.all_axes().values():
            axis.compile(axis_embedder)

    def add_persona_axis(self, axis: AxisSpec, axis_embedder: Any) -> None:
        axis.compile(axis_embedder)
        self.persona_axes[axis.name] = axis

    def merge_persona_axes(self, canonical_name: str, alias_name: str, *, note: str) -> None:
        if canonical_name == alias_name:
            return
        canonical = self.persona_axes.get(canonical_name)
        alias = self.persona_axes.get(alias_name)
        if canonical is None or alias is None:
            return
        merged_aliases = list(dict.fromkeys([*canonical.aliases, alias.name, alias.positive_pole, *alias.aliases]))
        canonical.aliases = tuple(str(item) for item in merged_aliases if item)
        canonical.support_count += max(1, alias.support_count)
        canonical.last_merged_ts = now_ts()
        self.axis_aliases[alias.name] = canonical.name
        self.axis_aliases[alias.positive_pole] = canonical.name
        self.merge_history.append(
            {
                "canonical": canonical.name,
                "alias": alias.name,
                "note": note,
                "ts": now_ts(),
            }
        )
        self.persona_axes.pop(alias_name, None)

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "profile_text": self.profile_text,
            "homeostatic_axes": {k: v.to_state_dict() for k, v in self.homeostatic_axes.items()},
            "persona_axes": {k: v.to_state_dict() for k, v in self.persona_axes.items()},
            "axis_aliases": dict(self.axis_aliases),
            "merge_history": list(self.merge_history),
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "PsychologicalSchema":
        return cls(
            homeostatic_axes={
                str(k): AxisSpec.from_state_dict(v)
                for k, v in data.get("homeostatic_axes", {}).items()
            },
            persona_axes={
                str(k): AxisSpec.from_state_dict(v)
                for k, v in data.get("persona_axes", {}).items()
            },
            profile_text=str(data.get("profile_text", "")),
            axis_aliases={str(k): str(v) for k, v in data.get("axis_aliases", {}).items()},
            merge_history=list(data.get("merge_history", [])),
        )


def schema_from_profile(
    *,
    axis_embedder: Any,
    profile_text: str = "",
    persona_axes: Optional[List[Dict[str, Any]]] = None,
) -> PsychologicalSchema:
    schema = PsychologicalSchema(profile_text=profile_text)
    for name, pos, neg, desc in HOMEOSTATIC_AXES:
        schema.homeostatic_axes[name] = AxisSpec(
            name=name,
            positive_pole=pos,
            negative_pole=neg,
            description=desc,
            level="homeostatic",
        )

    for axis_data in persona_axes or []:
        schema.persona_axes[str(axis_data["name"])] = AxisSpec(
            name=str(axis_data["name"]),
            positive_pole=str(axis_data.get("positive_pole", axis_data.get("positive", "more"))),
            negative_pole=str(axis_data.get("negative_pole", axis_data.get("negative", "less"))),
            description=str(axis_data.get("description", "")),
            level="persona",
            weight=float(axis_data.get("weight", 1.0)),
            positive_examples=tuple(str(item) for item in axis_data.get("positive_examples", [])),
            negative_examples=tuple(str(item) for item in axis_data.get("negative_examples", [])),
        )

    schema.compile(axis_embedder)
    return schema


def heuristic_persona_axes(profile_text: str) -> List[Dict[str, Any]]:
    text = profile_text.strip()
    if not text:
        return []

    fragments = re.split(r"[,\n;；，|、]+", text)
    seen: set[str] = set()
    axes: List[Dict[str, Any]] = []

    def simplify(fragment: str) -> str:
        value = fragment.strip()
        for hint in sorted(ANTONYM_HINTS.keys(), key=len, reverse=True):
            if hint in value.lower():
                return hint
        value = re.sub(r"^(很|更|太|比较|非常|想要|重视)", "", value)
        value = re.sub(r"(的人|的人格|的角色|的方式)$", "", value)
        return value[:18]

    for idx, raw in enumerate(fragments):
        chunk = raw.strip()
        if not chunk:
            continue
        match = re.search(r"(.+?)(?:/| vs |↔|->|=>| to )(.+)", chunk, flags=re.IGNORECASE)
        if match:
            pos = simplify(match.group(1))
            neg = simplify(match.group(2))
            name = re.sub(r"\W+", "_", pos.lower())[:24] or f"persona_{idx}"
        else:
            pos = simplify(chunk)
            neg = ANTONYM_HINTS.get(pos.lower(), ANTONYM_HINTS.get(pos, f"not_{pos}"))
            name = re.sub(r"\W+", "_", pos.lower())[:24] or f"persona_{idx}"
        if name in seen or not pos:
            continue
        seen.add(name)
        axes.append(
            {
                "name": name,
                "positive_pole": pos,
                "negative_pole": neg,
                "description": f"Persona axis derived from profile fragment: {chunk}",
                "positive_examples": tokenize_loose(chunk)[:4],
            }
        )
    return axes


@dataclass
class EventFrame:
    axis_evidence: Dict[str, float] = field(default_factory=dict)
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

    def axis_vector(self, axis_names: Sequence[str]) -> np.ndarray:
        return np.asarray([self.axis_evidence.get(name, 0.0) for name in axis_names], dtype=np.float32)

    def alignment_score(self, axis_state: Dict[str, float]) -> float:
        keys = [name for name in self.axis_evidence.keys() if name in axis_state]
        if not keys:
            return 0.0
        return float(np.mean([self.axis_evidence[name] * axis_state[name] for name in keys]))

    def strength(self) -> float:
        return float(
            0.45 * mean_abs(self.axis_evidence.values())
            + 0.20 * abs(self.valence)
            + 0.20 * self.arousal
            + 0.15 * self.self_relevance
        )

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "axis_evidence": {k: float(v) for k, v in self.axis_evidence.items()},
            "valence": float(self.valence),
            "arousal": float(self.arousal),
            "care": float(self.care),
            "threat": float(self.threat),
            "control": float(self.control),
            "abandonment": float(self.abandonment),
            "agency_signal": float(self.agency_signal),
            "shame": float(self.shame),
            "novelty": float(self.novelty),
            "self_relevance": float(self.self_relevance),
            "tags": list(self.tags),
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "EventFrame":
        return cls(
            axis_evidence={k: float(v) for k, v in data.get("axis_evidence", {}).items()},
            valence=float(data.get("valence", 0.0)),
            arousal=float(data.get("arousal", 0.0)),
            care=float(data.get("care", 0.0)),
            threat=float(data.get("threat", 0.0)),
            control=float(data.get("control", 0.0)),
            abandonment=float(data.get("abandonment", 0.0)),
            agency_signal=float(data.get("agency_signal", 0.0)),
            shame=float(data.get("shame", 0.0)),
            novelty=float(data.get("novelty", 0.0)),
            self_relevance=float(data.get("self_relevance", 0.5)),
            tags=tuple(data.get("tags", ())),
        )


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
    pred_error: float = 0.0
    redundancy: float = 0.0
    goal_relevance: float = 0.0
    contradiction: float = 0.0
    tension: float = 0.0
    story_id: Optional[str] = None
    theme_id: Optional[str] = None
    fact_keys: List[str] = field(default_factory=list)
    access_count: int = 0
    last_access_ts: float = field(default_factory=now_ts)
    status: Literal["active", "absorbed", "archived"] = "active"

    def mass(self) -> float:
        age = max(1.0, now_ts() - self.ts)
        freshness = 1.0 / math.log1p(age)
        source_factor = {
            "wake": 1.0,
            "repair": 0.9,
            "mode": 0.85,
            "dream": 0.65,
        }[self.source]
        return (
            freshness
            * (0.5 + math.log1p(self.access_count + 1))
            * (0.4 + self.confidence)
            * (0.5 + self.tension)
            * source_factor
        )

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "ts": float(self.ts),
            "text": self.text,
            "actors": list(self.actors),
            "embedding": vec_to_state(self.embedding),
            "frame": self.frame.to_state_dict(),
            "source": self.source,
            "confidence": float(self.confidence),
            "evidence_weight": float(self.evidence_weight),
            "surprise": float(self.surprise),
            "pred_error": float(self.pred_error),
            "redundancy": float(self.redundancy),
            "goal_relevance": float(self.goal_relevance),
            "contradiction": float(self.contradiction),
            "tension": float(self.tension),
            "story_id": self.story_id,
            "theme_id": self.theme_id,
            "fact_keys": list(self.fact_keys),
            "access_count": int(self.access_count),
            "last_access_ts": float(self.last_access_ts),
            "status": self.status,
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "Plot":
        embedding = vec_from_state(data.get("embedding"))
        if embedding is None:
            embedding = np.zeros(0, dtype=np.float32)
        return cls(
            id=str(data["id"]),
            ts=float(data["ts"]),
            text=str(data["text"]),
            actors=tuple(data.get("actors", ())),
            embedding=embedding,
            frame=EventFrame.from_state_dict(data.get("frame", {})),
            source=str(data.get("source", "wake")),
            confidence=float(data.get("confidence", 1.0)),
            evidence_weight=float(data.get("evidence_weight", 1.0)),
            surprise=float(data.get("surprise", 0.0)),
            pred_error=float(data.get("pred_error", 0.0)),
            redundancy=float(data.get("redundancy", 0.0)),
            goal_relevance=float(data.get("goal_relevance", 0.0)),
            contradiction=float(data.get("contradiction", 0.0)),
            tension=float(data.get("tension", 0.0)),
            story_id=data.get("story_id"),
            theme_id=data.get("theme_id"),
            fact_keys=[str(item) for item in data.get("fact_keys", [])],
            access_count=int(data.get("access_count", 0)),
            last_access_ts=float(data.get("last_access_ts", now_ts())),
            status=str(data.get("status", "active")),
        )


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
    unresolved_energy: float = 0.0
    status: Literal["active", "absorbed", "archived"] = "active"

    def _update_stats(self, name: str, x: float) -> None:
        if name == "dist":
            self.dist_n += 1
            delta = x - self.dist_mean
            self.dist_mean += delta / self.dist_n
            self.dist_m2 += delta * (x - self.dist_mean)
            return
        if name == "gap":
            self.gap_n += 1
            delta = x - self.gap_mean
            self.gap_mean += delta / self.gap_n
            self.gap_m2 += delta * (x - self.gap_mean)
            return
        raise ValueError(name)

    def dist_var(self) -> float:
        return self.dist_m2 / (self.dist_n - 1) if self.dist_n > 1 else 1.0

    def gap_mean_safe(self, default: float = 3600.0) -> float:
        return self.gap_mean if self.gap_n > 0 and self.gap_mean > 0 else default

    def mass(self) -> float:
        age = max(1.0, now_ts() - self.updated_ts)
        freshness = 1.0 / math.log1p(age)
        size = math.log1p(len(self.plot_ids) + 1)
        tension_bonus = 1.0 + 0.25 * mean_abs(self.tension_curve[-8:])
        return freshness * (size + math.log1p(self.reference_count + 1)) * tension_bonus

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_ts": float(self.created_ts),
            "updated_ts": float(self.updated_ts),
            "plot_ids": list(self.plot_ids),
            "centroid": vec_to_state(self.centroid),
            "actor_counts": {k: int(v) for k, v in self.actor_counts.items()},
            "tag_counts": {k: int(v) for k, v in self.tag_counts.items()},
            "source_counts": {k: int(v) for k, v in self.source_counts.items()},
            "tension_curve": [float(x) for x in self.tension_curve],
            "reference_count": int(self.reference_count),
            "dist_mean": float(self.dist_mean),
            "dist_m2": float(self.dist_m2),
            "dist_n": int(self.dist_n),
            "gap_mean": float(self.gap_mean),
            "gap_m2": float(self.gap_m2),
            "gap_n": int(self.gap_n),
            "unresolved_energy": float(self.unresolved_energy),
            "status": self.status,
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "StoryArc":
        return cls(
            id=str(data["id"]),
            created_ts=float(data["created_ts"]),
            updated_ts=float(data["updated_ts"]),
            plot_ids=list(data.get("plot_ids", [])),
            centroid=vec_from_state(data.get("centroid")),
            actor_counts={k: int(v) for k, v in data.get("actor_counts", {}).items()},
            tag_counts={k: int(v) for k, v in data.get("tag_counts", {}).items()},
            source_counts={k: int(v) for k, v in data.get("source_counts", {}).items()},
            tension_curve=[float(item) for item in data.get("tension_curve", [])],
            reference_count=int(data.get("reference_count", 0)),
            dist_mean=float(data.get("dist_mean", 0.0)),
            dist_m2=float(data.get("dist_m2", 0.0)),
            dist_n=int(data.get("dist_n", 0)),
            gap_mean=float(data.get("gap_mean", 0.0)),
            gap_m2=float(data.get("gap_m2", 0.0)),
            gap_n=int(data.get("gap_n", 0)),
            unresolved_energy=float(data.get("unresolved_energy", 0.0)),
            status=str(data.get("status", "active")),
        )


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
    name: str = ""
    description: str = ""
    theme_type: Literal["pattern", "lesson", "preference", "causality", "capability", "limitation", "self"] = "pattern"

    def confidence(self) -> float:
        return self.a / (self.a + self.b)

    def update_evidence(self, success: bool) -> None:
        if success:
            self.a += 1.0
        else:
            self.b += 1.0
        self.updated_ts = now_ts()

    def mass(self) -> float:
        age = max(1.0, now_ts() - self.updated_ts)
        freshness = 1.0 / math.log1p(age)
        return freshness * math.log1p(len(self.story_ids) + 1.0) * (0.5 + self.confidence())

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_ts": float(self.created_ts),
            "updated_ts": float(self.updated_ts),
            "story_ids": list(self.story_ids),
            "prototype": vec_to_state(self.prototype),
            "a": float(self.a),
            "b": float(self.b),
            "label": self.label,
            "name": self.name,
            "description": self.description,
            "theme_type": self.theme_type,
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "Theme":
        return cls(
            id=str(data["id"]),
            created_ts=float(data["created_ts"]),
            updated_ts=float(data["updated_ts"]),
            story_ids=list(data.get("story_ids", [])),
            prototype=vec_from_state(data.get("prototype")),
            a=float(data.get("a", 1.0)),
            b=float(data.get("b", 1.0)),
            label=str(data.get("label", "")),
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            theme_type=str(data.get("theme_type", "pattern")),
        )


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

    def score(self, self_vector: np.ndarray, axis_state: Dict[str, float]) -> float:
        semantic = cosine_sim(self_vector, self.prototype)
        common = [name for name in self.axis_prototype.keys() if name in axis_state]
        if not common:
            return semantic
        axis_score = float(np.mean([1.0 - abs(axis_state[name] - self.axis_prototype[name]) for name in common]))
        return 0.65 * semantic + 0.35 * axis_score

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "prototype": vec_to_state(self.prototype),
            "axis_prototype": {k: float(v) for k, v in self.axis_prototype.items()},
            "support": int(self.support),
            "barrier": float(self.barrier),
            "hysteresis": float(self.hysteresis),
            "created_ts": float(self.created_ts),
            "updated_ts": float(self.updated_ts),
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "IdentityMode":
        prototype = vec_from_state(data.get("prototype"))
        if prototype is None:
            prototype = np.zeros(0, dtype=np.float32)
        return cls(
            id=str(data["id"]),
            label=str(data["label"]),
            prototype=prototype,
            axis_prototype={k: float(v) for k, v in data.get("axis_prototype", {}).items()},
            support=int(data.get("support", 1)),
            barrier=float(data.get("barrier", 0.55)),
            hysteresis=float(data.get("hysteresis", 0.08)),
            created_ts=float(data.get("created_ts", now_ts())),
            updated_ts=float(data.get("updated_ts", now_ts())),
        )


@dataclass
class IdentityState:
    self_vector: np.ndarray
    axis_state: Dict[str, float] = field(default_factory=dict)
    intuition_axes: Dict[str, float] = field(default_factory=dict)
    active_energy: float = 0.0
    repressed_energy: float = 0.0
    contradiction_ema: float = 0.0
    current_mode_id: Optional[str] = None
    current_mode_label: str = "origin"
    repair_count: int = 0
    dream_count: int = 0
    mode_change_count: int = 0
    narrative_log: List[str] = field(default_factory=list)
    last_mode_step: int = 0
    last_mode_change_ts: float = field(default_factory=now_ts)

    def plasticity(self) -> float:
        openness = (self.axis_state.get("exploration", 0.0) + 1.0) / 2.0
        coherence = (self.axis_state.get("coherence", 0.0) + 1.0) / 2.0
        regulation = (self.axis_state.get("regulation", 0.0) + 1.0) / 2.0
        vigilance = (self.axis_state.get("vigilance", 0.0) + 1.0) / 2.0
        return clamp01(0.35 * openness + 0.35 * coherence + 0.20 * regulation + 0.10 * (1.0 - vigilance))

    def rigidity(self) -> float:
        openness = (self.axis_state.get("exploration", 0.0) + 1.0) / 2.0
        vigilance = (self.axis_state.get("vigilance", 0.0) + 1.0) / 2.0
        regulation = (self.axis_state.get("regulation", 0.0) + 1.0) / 2.0
        return clamp01(0.45 * vigilance + 0.35 * (1.0 - openness) + 0.20 * (1.0 - regulation))

    def narrative_pressure(self) -> float:
        return float(self.active_energy + 0.85 * self.repressed_energy + 1.20 * self.contradiction_ema)

    def axis_vector(self, axis_names: Sequence[str]) -> np.ndarray:
        return np.asarray([self.axis_state.get(name, 0.0) for name in axis_names], dtype=np.float32)

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "self_vector": vec_to_state(self.self_vector),
            "axis_state": {k: float(v) for k, v in self.axis_state.items()},
            "intuition_axes": {k: float(v) for k, v in self.intuition_axes.items()},
            "active_energy": float(self.active_energy),
            "repressed_energy": float(self.repressed_energy),
            "contradiction_ema": float(self.contradiction_ema),
            "current_mode_id": self.current_mode_id,
            "current_mode_label": self.current_mode_label,
            "repair_count": int(self.repair_count),
            "dream_count": int(self.dream_count),
            "mode_change_count": int(self.mode_change_count),
            "narrative_log": list(self.narrative_log),
            "last_mode_step": int(self.last_mode_step),
            "last_mode_change_ts": float(self.last_mode_change_ts),
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "IdentityState":
        self_vector = vec_from_state(data.get("self_vector"))
        if self_vector is None:
            self_vector = np.zeros(0, dtype=np.float32)
        return cls(
            self_vector=self_vector,
            axis_state={k: float(v) for k, v in data.get("axis_state", {}).items()},
            intuition_axes={k: float(v) for k, v in data.get("intuition_axes", {}).items()},
            active_energy=float(data.get("active_energy", 0.0)),
            repressed_energy=float(data.get("repressed_energy", 0.0)),
            contradiction_ema=float(data.get("contradiction_ema", 0.0)),
            current_mode_id=data.get("current_mode_id"),
            current_mode_label=str(data.get("current_mode_label", "origin")),
            repair_count=int(data.get("repair_count", 0)),
            dream_count=int(data.get("dream_count", 0)),
            mode_change_count=int(data.get("mode_change_count", 0)),
            narrative_log=list(data.get("narrative_log", [])),
            last_mode_step=int(data.get("last_mode_step", 0)),
            last_mode_change_ts=float(data.get("last_mode_change_ts", now_ts())),
        )


@dataclass(frozen=True)
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

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "plot_id": self.plot_id,
            "ts": float(self.ts),
            "activation": float(self.activation),
            "unresolved": float(self.unresolved),
            "embedding": vec_to_state(self.embedding),
            "axis_evidence": {k: float(v) for k, v in self.axis_evidence.items()},
            "source": self.source,
            "tags": list(self.tags),
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "LatentFragment":
        embedding = vec_from_state(data.get("embedding"))
        if embedding is None:
            embedding = np.zeros(0, dtype=np.float32)
        return cls(
            plot_id=str(data["plot_id"]),
            ts=float(data["ts"]),
            activation=float(data.get("activation", 0.0)),
            unresolved=float(data.get("unresolved", 0.0)),
            embedding=embedding,
            axis_evidence={k: float(v) for k, v in data.get("axis_evidence", {}).items()},
            source=str(data.get("source", "wake")),
            tags=tuple(data.get("tags", ())),
        )


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

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "edge_type": self.edge_type,
            "a": float(self.a),
            "b": float(self.b),
            "use_count": int(self.use_count),
            "last_used_ts": float(self.last_used_ts),
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "EdgeBelief":
        return cls(
            edge_type=str(data["edge_type"]),
            a=float(data.get("a", 1.0)),
            b=float(data.get("b", 1.0)),
            use_count=int(data.get("use_count", 0)),
            last_used_ts=float(data.get("last_used_ts", now_ts())),
        )


@dataclass
class RetrievalTrace:
    query: str
    query_emb: np.ndarray
    attractor_path: List[np.ndarray]
    ranked: List[Tuple[str, float, str]]
    query_type: Optional["QueryType"] = None
    time_range: Optional["TimeRange"] = None


@dataclass(frozen=True)
class IdentitySnapshot:
    current_mode: str
    axis_state: Dict[str, float]
    intuition_axes: Dict[str, float]
    persona_axes: Dict[str, Dict[str, Any]]
    axis_aliases: Dict[str, str]
    modes: Dict[str, Dict[str, Any]]
    active_energy: float
    repressed_energy: float
    contradiction_ema: float
    plasticity: float
    rigidity: float
    repair_count: int
    dream_count: int
    mode_change_count: int
    narrative_tail: List[str]

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "current_mode": self.current_mode,
            "axis_state": {k: float(v) for k, v in self.axis_state.items()},
            "intuition_axes": {k: float(v) for k, v in self.intuition_axes.items()},
            "persona_axes": self.persona_axes,
            "axis_aliases": dict(self.axis_aliases),
            "modes": self.modes,
            "active_energy": float(self.active_energy),
            "repressed_energy": float(self.repressed_energy),
            "contradiction_ema": float(self.contradiction_ema),
            "plasticity": float(self.plasticity),
            "rigidity": float(self.rigidity),
            "repair_count": int(self.repair_count),
            "dream_count": int(self.dream_count),
            "mode_change_count": int(self.mode_change_count),
            "narrative_tail": list(self.narrative_tail),
        }


@dataclass(frozen=True)
class NarrativeSummary:
    text: str
    current_mode: str
    pressure: float
    salient_axes: List[str]

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "current_mode": self.current_mode,
            "pressure": float(self.pressure),
            "salient_axes": list(self.salient_axes),
        }


def top_axes_description(axis_state: Dict[str, float], schema: PsychologicalSchema, topn: int = 4) -> str:
    items = sorted(axis_state.items(), key=lambda item: abs(item[1]), reverse=True)
    phrases = []
    for name, value in items[:topn]:
        axis = schema.all_axes().get(name)
        if axis is None:
            continue
        pole = axis.positive_pole if value >= 0 else axis.negative_pole
        phrases.append(f"{pole}({value:+.2f})")
    return " / ".join(phrases) if phrases else "unformed"


def axes_to_phrase(axis_names: Sequence[str], schema: PsychologicalSchema) -> str:
    phrases = []
    for name in axis_names:
        axis = schema.all_axes().get(name)
        if axis is None:
            phrases.append(name)
            continue
        phrases.append(f"{axis.positive_pole}/{axis.negative_pole}")
    return " / ".join(phrases) if phrases else "unnamed tension"
