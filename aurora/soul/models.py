from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np

from aurora.utils.math_utils import cosine_sim, l2_normalize, sigmoid
from aurora.utils.time_utils import now_ts

if TYPE_CHECKING:
    from aurora.soul.query import QueryType, TimeRange

TRAIT_ORDER: Tuple[str, ...] = (
    "attachment",
    "autonomy",
    "trust",
    "vigilance",
    "openness",
    "defensiveness",
    "assertiveness",
    "coherence",
)

BELIEF_ORDER: Tuple[str, ...] = (
    "closeness_safe",
    "others_reliable",
    "boundaries_allowed",
    "independence_safe",
    "vulnerability_safe",
)


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def signed01(x: float) -> float:
    return 2.0 * clamp(x) - 1.0


def unsign11(x: float) -> float:
    return clamp((float(x) + 1.0) / 2.0)


def softmax(logits: Sequence[float], temperature: float = 1.0) -> List[float]:
    t = max(float(temperature), 1e-6)
    scaled = [float(x) / t for x in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    total = sum(exps) + 1e-12
    return [x / total for x in exps]


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


@dataclass
class EventFrame:
    trait_evidence: Dict[str, float] = field(default_factory=dict)
    belief_evidence: Dict[str, float] = field(default_factory=dict)
    valence: float = 0.0
    arousal: float = 0.0
    tags: Tuple[str, ...] = ()
    threat: float = 0.0
    care: float = 0.0
    control: float = 0.0
    abandonment: float = 0.0
    agency: float = 0.0
    shame: float = 0.0

    def trait_vector(self, order: Sequence[str] = TRAIT_ORDER) -> np.ndarray:
        return np.asarray([self.trait_evidence.get(name, 0.0) for name in order], dtype=np.float32)

    def belief_vector(self, order: Sequence[str] = BELIEF_ORDER) -> np.ndarray:
        return np.asarray([self.belief_evidence.get(name, 0.0) for name in order], dtype=np.float32)

    def strength(self) -> float:
        return float(
            0.35 * mean_abs(self.trait_evidence.values())
            + 0.25 * mean_abs(self.belief_evidence.values())
            + 0.20 * abs(self.valence)
            + 0.20 * self.arousal
        )

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "trait_evidence": {k: float(v) for k, v in self.trait_evidence.items()},
            "belief_evidence": {k: float(v) for k, v in self.belief_evidence.items()},
            "valence": float(self.valence),
            "arousal": float(self.arousal),
            "tags": list(self.tags),
            "threat": float(self.threat),
            "care": float(self.care),
            "control": float(self.control),
            "abandonment": float(self.abandonment),
            "agency": float(self.agency),
            "shame": float(self.shame),
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "EventFrame":
        return cls(
            trait_evidence={k: float(v) for k, v in data.get("trait_evidence", {}).items()},
            belief_evidence={k: float(v) for k, v in data.get("belief_evidence", {}).items()},
            valence=float(data.get("valence", 0.0)),
            arousal=float(data.get("arousal", 0.0)),
            tags=tuple(data.get("tags", ())),
            threat=float(data.get("threat", 0.0)),
            care=float(data.get("care", 0.0)),
            control=float(data.get("control", 0.0)),
            abandonment=float(data.get("abandonment", 0.0)),
            agency=float(data.get("agency", 0.0)),
            shame=float(data.get("shame", 0.0)),
        )


@dataclass
class Plot:
    id: str
    ts: float
    text: str
    actors: Tuple[str, ...]
    embedding: np.ndarray
    frame: EventFrame
    source: Literal["wake", "dream", "repair", "phase"] = "wake"
    confidence: float = 1.0
    evidence_weight: float = 1.0
    surprise: float = 0.0
    pred_error: float = 0.0
    redundancy: float = 0.0
    goal_relevance: float = 0.0
    contradiction: float = 0.0
    tension: float = 0.0
    story_id: Optional[str] = None
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
            "phase": 0.85,
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
    dist_mean: float = 0.0
    dist_m2: float = 0.0
    dist_n: int = 0
    gap_mean: float = 0.0
    gap_m2: float = 0.0
    gap_n: int = 0
    actor_counts: Dict[str, int] = field(default_factory=dict)
    tension_curve: List[float] = field(default_factory=list)
    source_counts: Dict[str, int] = field(default_factory=dict)
    unresolved_energy: float = 0.0
    status: Literal["developing", "resolved", "abandoned"] = "developing"
    reference_count: int = 0

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

    def activity_probability(self, ts: Optional[float] = None) -> float:
        ts = ts or now_ts()
        idle = max(0.0, ts - self.updated_ts)
        tau = self.gap_mean_safe()
        return math.exp(-idle / max(tau, 1e-6))

    def source_purity(self) -> float:
        total = sum(self.source_counts.values())
        if total <= 0:
            return 1.0
        return max(self.source_counts.values()) / total

    def mass(self) -> float:
        age = max(1.0, now_ts() - self.updated_ts)
        freshness = 1.0 / math.log1p(age)
        size = math.log1p(len(self.plot_ids))
        tension_bonus = 1.0 + 0.25 * mean_abs(self.tension_curve[-8:])
        return freshness * (size + math.log1p(self.reference_count + 1)) * tension_bonus

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_ts": float(self.created_ts),
            "updated_ts": float(self.updated_ts),
            "plot_ids": list(self.plot_ids),
            "centroid": vec_to_state(self.centroid),
            "dist_mean": float(self.dist_mean),
            "dist_m2": float(self.dist_m2),
            "dist_n": int(self.dist_n),
            "gap_mean": float(self.gap_mean),
            "gap_m2": float(self.gap_m2),
            "gap_n": int(self.gap_n),
            "actor_counts": {k: int(v) for k, v in self.actor_counts.items()},
            "tension_curve": [float(x) for x in self.tension_curve],
            "source_counts": {k: int(v) for k, v in self.source_counts.items()},
            "unresolved_energy": float(self.unresolved_energy),
            "status": self.status,
            "reference_count": int(self.reference_count),
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "StoryArc":
        return cls(
            id=str(data["id"]),
            created_ts=float(data["created_ts"]),
            updated_ts=float(data["updated_ts"]),
            plot_ids=list(data.get("plot_ids", [])),
            centroid=vec_from_state(data.get("centroid")),
            dist_mean=float(data.get("dist_mean", 0.0)),
            dist_m2=float(data.get("dist_m2", 0.0)),
            dist_n=int(data.get("dist_n", 0)),
            gap_mean=float(data.get("gap_mean", 0.0)),
            gap_m2=float(data.get("gap_m2", 0.0)),
            gap_n=int(data.get("gap_n", 0)),
            actor_counts={k: int(v) for k, v in data.get("actor_counts", {}).items()},
            tension_curve=[float(x) for x in data.get("tension_curve", [])],
            source_counts={k: int(v) for k, v in data.get("source_counts", {}).items()},
            unresolved_energy=float(data.get("unresolved_energy", 0.0)),
            status=str(data.get("status", "developing")),
            reference_count=int(data.get("reference_count", 0)),
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
        return freshness * math.log1p(len(self.story_ids) + 1) * (0.5 + self.confidence())

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_ts": float(self.created_ts),
            "updated_ts": float(self.updated_ts),
            "story_ids": list(self.story_ids),
            "prototype": vec_to_state(self.prototype),
            "a": float(self.a),
            "b": float(self.b),
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
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            theme_type=str(data.get("theme_type", "pattern")),
        )


@dataclass
class DissonanceReport:
    trait_conflicts: Dict[str, float]
    belief_conflicts: Dict[str, float]
    trait_alignment: Dict[str, float]
    belief_alignment: Dict[str, float]
    affective_load: float
    total: float


@dataclass
class IdentityState:
    traits: Dict[str, float]
    beliefs: Dict[str, float]
    phase: str
    active_energy: float = 0.0
    repressed_energy: float = 0.0
    contradiction_ema: float = 0.0
    intuition: Dict[str, float] = field(default_factory=lambda: {t: 0.0 for t in TRAIT_ORDER})
    repair_count: int = 0
    dream_count: int = 0
    narrative_log: List[str] = field(default_factory=list)
    last_phase_step: int = 0
    last_phase_change_ts: float = field(default_factory=now_ts)

    def signed_traits(self, order: Sequence[str] = TRAIT_ORDER) -> np.ndarray:
        return np.asarray([signed01(self.traits[name]) for name in order], dtype=np.float32)

    def signed_beliefs(self, order: Sequence[str] = BELIEF_ORDER) -> np.ndarray:
        return np.asarray([signed01(self.beliefs[name]) for name in order], dtype=np.float32)

    def plasticity(self) -> float:
        return clamp(
            0.45 * self.traits["openness"]
            + 0.30 * self.traits["coherence"]
            + 0.25 * (1.0 - self.traits["defensiveness"])
        )

    def rigidity(self) -> float:
        return clamp(
            0.45 * self.traits["defensiveness"]
            + 0.25 * self.traits["vigilance"]
            + 0.30 * (1.0 - self.traits["openness"])
        )

    def narrative_pressure(self) -> float:
        return float(self.active_energy + 0.8 * self.repressed_energy + 1.25 * self.contradiction_ema)

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "traits": {k: float(v) for k, v in self.traits.items()},
            "beliefs": {k: float(v) for k, v in self.beliefs.items()},
            "phase": self.phase,
            "active_energy": float(self.active_energy),
            "repressed_energy": float(self.repressed_energy),
            "contradiction_ema": float(self.contradiction_ema),
            "intuition": {k: float(v) for k, v in self.intuition.items()},
            "repair_count": int(self.repair_count),
            "dream_count": int(self.dream_count),
            "narrative_log": list(self.narrative_log),
            "last_phase_step": int(self.last_phase_step),
            "last_phase_change_ts": float(self.last_phase_change_ts),
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "IdentityState":
        return cls(
            traits={k: float(v) for k, v in data.get("traits", {}).items()},
            beliefs={k: float(v) for k, v in data.get("beliefs", {}).items()},
            phase=str(data.get("phase", "dependent_child")),
            active_energy=float(data.get("active_energy", 0.0)),
            repressed_energy=float(data.get("repressed_energy", 0.0)),
            contradiction_ema=float(data.get("contradiction_ema", 0.0)),
            intuition={k: float(v) for k, v in data.get("intuition", {}).items()},
            repair_count=int(data.get("repair_count", 0)),
            dream_count=int(data.get("dream_count", 0)),
            narrative_log=list(data.get("narrative_log", [])),
            last_phase_step=int(data.get("last_phase_step", 0)),
            last_phase_change_ts=float(data.get("last_phase_change_ts", now_ts())),
        )


@dataclass(frozen=True)
class PhaseProfile:
    name: str
    prototype: Dict[str, float]
    barrier: float
    hysteresis: float
    description: str


@dataclass(frozen=True)
class CandidateRepair:
    mode: str
    new_traits: Dict[str, float]
    new_beliefs: Dict[str, float]
    active_energy_after: float
    repressed_energy_after: float
    identity_drift: float
    coherence_gain: float
    utility: float
    explanation: str


@dataclass
class LatentFragment:
    plot_id: str
    ts: float
    activation: float
    unresolved: float
    trait_vector: np.ndarray
    source: str
    tags: Tuple[str, ...]

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "plot_id": self.plot_id,
            "ts": float(self.ts),
            "activation": float(self.activation),
            "unresolved": float(self.unresolved),
            "trait_vector": vec_to_state(self.trait_vector),
            "source": self.source,
            "tags": list(self.tags),
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "LatentFragment":
        trait_vector = vec_from_state(data.get("trait_vector"))
        if trait_vector is None:
            trait_vector = np.zeros(len(TRAIT_ORDER), dtype=np.float32)
        return cls(
            plot_id=str(data["plot_id"]),
            ts=float(data["ts"]),
            activation=float(data.get("activation", 0.0)),
            unresolved=float(data.get("unresolved", 0.0)),
            trait_vector=trait_vector,
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
    phase: str
    traits: Dict[str, float]
    beliefs: Dict[str, float]
    active_energy: float
    repressed_energy: float
    contradiction_ema: float
    plasticity: float
    rigidity: float
    intuition: Dict[str, float]
    repair_count: int
    dream_count: int
    narrative_tail: List[str]

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "traits": {k: float(v) for k, v in self.traits.items()},
            "beliefs": {k: float(v) for k, v in self.beliefs.items()},
            "active_energy": float(self.active_energy),
            "repressed_energy": float(self.repressed_energy),
            "contradiction_ema": float(self.contradiction_ema),
            "plasticity": float(self.plasticity),
            "rigidity": float(self.rigidity),
            "intuition": {k: float(v) for k, v in self.intuition.items()},
            "repair_count": int(self.repair_count),
            "dream_count": int(self.dream_count),
            "narrative_tail": list(self.narrative_tail),
        }


@dataclass(frozen=True)
class NarrativeSummary:
    text: str
    phase: str
    phase_description: str
    core_statement: str
    pressure: float

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "phase": self.phase,
            "phase_description": self.phase_description,
            "core_statement": self.core_statement,
            "pressure": float(self.pressure),
        }


def summarize_identity(identity: IdentityState, profile: PhaseProfile) -> NarrativeSummary:
    if identity.phase == "dependent_child":
        core = "她仍然更相信靠近与被接住。"
    elif identity.phase == "guarded_teen":
        core = "她已经学会先保护自己，亲近不再自动等于安全。"
    elif identity.phase == "exploratory_youth":
        core = "她在独立与连接之间试探新的平衡。"
    else:
        core = "她开始把柔软、边界和自主整合进同一个自己。"

    line2 = (
        f"信任={identity.traits['trust']:.2f}，自主={identity.traits['autonomy']:.2f}，"
        f"防御={identity.traits['defensiveness']:.2f}，连贯={identity.traits['coherence']:.2f}。"
    )
    line3 = (
        f"当前叙事压力={identity.narrative_pressure():.3f}，"
        f"显性能量={identity.active_energy:.3f}，压抑能量={identity.repressed_energy:.3f}。"
    )
    text = f"{profile.description} {core} {line2} {line3}"
    return NarrativeSummary(
        text=text,
        phase=identity.phase,
        phase_description=profile.description,
        core_statement=core,
        pressure=identity.narrative_pressure(),
    )
