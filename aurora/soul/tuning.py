"""Centralized Aurora V7 tuning models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional

from aurora.utils.jsonx import loads


def _merge_nested(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _validate_weight_sum(name: str, values: Mapping[str, float], *, tol: float = 1e-6) -> None:
    total = sum(float(value) for value in values.values())
    if abs(total - 1.0) > tol:
        raise ValueError(f"{name} must sum to 1.0, got {total:.6f}")


@dataclass(frozen=True)
class AffectiveLoadWeights:
    arousal: float = 0.30
    negative_valence: float = 0.20
    threat: float = 0.18
    shame: float = 0.12
    control: float = 0.10
    abandonment: float = 0.10

    def validate(self) -> None:
        _validate_weight_sum("affective_load_weights", asdict(self))


@dataclass(frozen=True)
class DissonanceWeights:
    axis_conflict: float = 0.36
    semantic_conflict: float = 0.20
    affective_load: float = 0.28
    narrative_incongruity: float = 0.16

    def validate(self) -> None:
        _validate_weight_sum("dissonance_weights", asdict(self))


@dataclass(frozen=True)
class TensionWeights:
    surprise: float = 0.30
    pred_error: float = 0.20
    contradiction: float = 0.20
    arousal: float = 0.15
    self_relevance: float = 0.10
    novelty: float = 0.05

    def validate(self) -> None:
        _validate_weight_sum("tension_weights", asdict(self))


@dataclass(frozen=True)
class RetrievalWeights:
    attractor_weight: float = 0.50
    direct_damping: float = 0.80
    pagerank_damping: float = 0.85
    negative_anchor_penalty: float = 0.14
    negative_candidate_penalty: float = 0.08
    plot_alignment_bonus: float = 0.15
    attractor_plot_alignment_bonus: float = 0.10
    attractor_plot_confidence_bonus: float = 0.03
    story_theme_identity_bonus: float = 0.05


@dataclass(frozen=True)
class GraphThresholds:
    similarity_threshold: float = 0.20
    contradiction_threshold: float = 0.16
    anchor_similarity_factor: float = 0.50
    min_edge_weight: float = 0.05
    temporal_edge_weight: float = 0.80
    summary_edge_weight: float = 0.50
    dream_integration_threshold: float = 0.35
    cold_plot_tension_max: float = 0.40
    cold_plot_contradiction_max: float = 0.35
    hot_plot_tension_min: float = 0.75


@dataclass(frozen=True)
class AssimilationTuning:
    persona_axis_rate: float = 0.015
    homeostatic_axis_rate: float = 0.025
    evidence_floor: float = 0.06
    self_relevance_base: float = 0.55
    self_relevance_bonus: float = 0.45
    contradiction_ema_rate: float = 0.24
    regulation_care_bonus: float = 0.03
    regulation_arousal_penalty: float = 0.04
    regulation_threat_penalty: float = 0.03
    vigilance_threat_bonus: float = 0.03
    vigilance_care_penalty: float = 0.02
    coherence_shame_penalty: float = 0.02
    coherence_care_bonus: float = 0.01


@dataclass(frozen=True)
class ViewRefreshTuning:
    plot_vector_semantic_weight: float = 0.72
    plot_vector_implication_weight: float = 0.28
    generated_dream_confidence: float = 0.34
    generated_repair_confidence: float = 0.72
    generated_dream_evidence_weight: float = 0.20
    generated_repair_evidence_weight: float = 0.35
    wake_energy_scale: float = 1.00
    dream_energy_scale: float = 0.35
    repair_energy_scale: float = 0.20
    mode_energy_scale: float = 0.15

    def validate(self) -> None:
        _validate_weight_sum(
            "plot_vector_mixing",
            {
                "semantic": self.plot_vector_semantic_weight,
                "implication": self.plot_vector_implication_weight,
            },
        )


@dataclass(frozen=True)
class AuroraTuning:
    affective_load: AffectiveLoadWeights = field(default_factory=AffectiveLoadWeights)
    dissonance: DissonanceWeights = field(default_factory=DissonanceWeights)
    tension: TensionWeights = field(default_factory=TensionWeights)
    retrieval: RetrievalWeights = field(default_factory=RetrievalWeights)
    graph: GraphThresholds = field(default_factory=GraphThresholds)
    assimilation: AssimilationTuning = field(default_factory=AssimilationTuning)
    view_refresh: ViewRefreshTuning = field(default_factory=ViewRefreshTuning)

    def validate(self) -> None:
        self.affective_load.validate()
        self.dissonance.validate()
        self.tension.validate()
        self.view_refresh.validate()

    def to_state_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, data: Optional[Mapping[str, Any]] = None) -> "AuroraTuning":
        base = cls().to_state_dict()
        merged = base if data is None else _merge_nested(base, data)
        tuning = cls(
            affective_load=AffectiveLoadWeights(**merged["affective_load"]),
            dissonance=DissonanceWeights(**merged["dissonance"]),
            tension=TensionWeights(**merged["tension"]),
            retrieval=RetrievalWeights(**merged["retrieval"]),
            graph=GraphThresholds(**merged["graph"]),
            assimilation=AssimilationTuning(**merged["assimilation"]),
            view_refresh=ViewRefreshTuning(**merged["view_refresh"]),
        )
        tuning.validate()
        return tuning

    @classmethod
    def from_json_path(cls, path: Optional[str]) -> "AuroraTuning":
        if not path:
            tuning = cls()
            tuning.validate()
            return tuning
        with open(path, "r", encoding="utf-8") as handle:
            payload = loads(handle.read())
        if not isinstance(payload, Mapping):
            raise ValueError("Aurora tuning file must contain a JSON object")
        return cls.from_state_dict(payload)
