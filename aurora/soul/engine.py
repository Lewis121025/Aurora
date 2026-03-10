from __future__ import annotations

import copy
import json
import math
import random
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from aurora.integrations.embeddings.base import EmbeddingProvider
from aurora.integrations.embeddings.local_semantic import LocalSemanticEmbedding
from aurora.soul.extractors import (
    CombinatorialNarrativeProvider,
    HeuristicMeaningProvider,
    MeaningProvider,
    NarrativeProvider,
)
from aurora.soul.facts import FactExtractor
from aurora.soul.models import (
    AxisSpec,
    DissonanceReport,
    EventFrame,
    IdentityMode,
    IdentitySnapshot,
    IdentityState,
    LatentFragment,
    NarrativeSummary,
    Plot,
    PsychologicalSchema,
    RepairCandidate,
    StoryArc,
    Theme,
    axes_to_phrase,
    clamp,
    clamp01,
    heuristic_persona_axes,
    l2_normalize,
    mean_abs,
    schema_from_profile,
    sigmoid,
    softmax,
    stable_hash,
    top_axes_description,
)
from aurora.soul.retrieval import (
    CRPAssigner,
    FieldRetriever,
    LowRankMetric,
    MemoryGraph,
    OnlineKDE,
    StoryModel,
    ThemeModel,
    ThompsonBernoulliGate,
    VectorIndex,
)
from aurora.utils.math_utils import cosine_sim
from aurora.utils.time_utils import now_ts

SOUL_MEMORY_STATE_VERSION = "aurora-soul-memory-v4"


def moving_average(old: float, new: float, rate: float) -> float:
    return (1.0 - rate) * float(old) + rate * float(new)


def stable_uuid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def weighted_choice(items: Sequence[Any], weights: Sequence[float], rng: np.random.Generator) -> Any:
    if not items:
        raise ValueError("weighted_choice on empty items")
    values = np.asarray([max(0.0, float(weight)) for weight in weights], dtype=np.float64)
    if float(values.sum()) <= 1e-12:
        return items[int(rng.integers(0, len(items)))]
    values /= values.sum()
    return items[int(rng.choice(np.arange(len(items)), p=values))]


class SubconsciousField:
    def __init__(self, reservoir: int = 1024, seed: int = 0):
        self.reservoir = reservoir
        self._seed = seed
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
        weights = []
        for fragment in self.fragments:
            age = max(1.0, now_ts() - fragment.ts)
            recency = 1.0 / math.log1p(age)
            weight = 0.55 * fragment.unresolved + 0.25 * fragment.activation + 0.20 * recency
            weights.append(max(1e-6, weight))
        pool = list(range(len(self.fragments)))
        chosen: List[LatentFragment] = []
        for _ in range(min(n, len(self.fragments))):
            idx = weighted_choice(pool, [weights[item] for item in pool], self.rng)
            chosen.append(self.fragments[idx])
            pool.remove(idx)
        return chosen

    def operator(self, fragments: Sequence[LatentFragment], state: IdentityState) -> str:
        unresolved = float(np.mean([fragment.unresolved for fragment in fragments])) if fragments else 0.0
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
        fragments: Sequence[LatentFragment],
        state: IdentityState,
        schema: PsychologicalSchema,
        narrator: NarrativeProvider,
    ) -> Tuple[str, EventFrame, float]:
        if not fragments:
            return (
                "A dream with unfinished words and no stable shape.",
                EventFrame(tags=("empty_dream",), arousal=0.05, self_relevance=0.8),
                0.0,
            )

        operator = self.operator(fragments, state)
        tags: List[str] = []
        for fragment in fragments:
            for tag in fragment.tags:
                if tag not in tags:
                    tags.append(tag)

        text = narrator.compose_dream(operator, tags, state, schema)

        blended = {name: 0.0 for name in schema.ordered_axis_names()}
        for fragment in fragments:
            for name, value in fragment.axis_evidence.items():
                blended[name] = blended.get(name, 0.0) + value / max(len(fragments), 1)

        if operator == "counterfactual":
            blended["agency"] = clamp(blended.get("agency", 0.0) + 0.18)
            blended["coherence"] = clamp(blended.get("coherence", 0.0) + 0.12)
        elif operator == "fear_rehearsal":
            blended["vigilance"] = clamp(blended.get("vigilance", 0.0) + 0.18)
            blended["regulation"] = clamp(blended.get("regulation", 0.0) - 0.08)
        elif operator == "wish_rehearsal":
            blended["affiliation"] = clamp(blended.get("affiliation", 0.0) + 0.12)
            blended["regulation"] = clamp(blended.get("regulation", 0.0) + 0.10)
        else:
            blended["coherence"] = clamp(blended.get("coherence", 0.0) + 0.18)
            blended["regulation"] = clamp(blended.get("regulation", 0.0) + 0.10)

        care = max(0.0, np.mean([0.25 + 0.25 * max(0.0, fragment.axis_evidence.get("affiliation", 0.0)) for fragment in fragments]))
        threat = max(0.0, np.mean([0.25 + 0.25 * max(0.0, fragment.axis_evidence.get("vigilance", 0.0)) for fragment in fragments]))
        valence = clamp(np.mean([sum(fragment.axis_evidence.values()) / max(len(fragment.axis_evidence), 1) for fragment in fragments]))
        arousal = clamp01(0.30 + 0.25 * np.mean([fragment.unresolved for fragment in fragments]))

        dream_frame = EventFrame(
            axis_evidence=blended,
            valence=valence,
            arousal=arousal,
            care=care,
            threat=threat,
            agency_signal=max(0.0, blended.get("agency", 0.0)),
            shame=max(0.0, -valence) * 0.15,
            novelty=0.40,
            self_relevance=0.85,
            tags=tuple(["dream", operator, *tags][:12]),
        )
        resonance = float(np.mean([np.dot(state.self_vector, fragment.embedding) * (0.5 + fragment.unresolved) for fragment in fragments]))
        return text, dream_frame, resonance

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "reservoir": self.reservoir,
            "seed": self._seed,
            "fragments": [fragment.to_state_dict() for fragment in self.fragments],
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "SubconsciousField":
        obj = cls(reservoir=int(data.get("reservoir", 1024)), seed=int(data.get("seed", 0)))
        obj.fragments = [LatentFragment.from_state_dict(item) for item in data.get("fragments", [])]
        return obj


class SchemaConsolidator:
    def __init__(self, every_events: int = 50, budget: int = 24):
        self.every_events = every_events
        self.budget = budget
        self.events_since_last = 0
        self.axis_texts: Dict[str, List[str]] = {}

    def observe(self, plot: Plot) -> None:
        for axis_name, value in plot.frame.axis_evidence.items():
            if abs(value) < 0.24:
                continue
            self.axis_texts.setdefault(axis_name, []).append(plot.text[:140])
            self.axis_texts[axis_name] = self.axis_texts[axis_name][-12:]
        self.events_since_last += 1

    def should_run(self, added_axis: bool, persona_count: int) -> bool:
        return added_axis or self.events_since_last >= self.every_events or persona_count > self.budget

    def maybe_consolidate(
        self,
        *,
        schema: PsychologicalSchema,
        axis_embedder: EmbeddingProvider,
        narrator: NarrativeProvider,
    ) -> List[Tuple[str, str]]:
        merges: List[Tuple[str, str]] = []
        if len(schema.persona_axes) < 2:
            self.events_since_last = 0
            return merges

        while True:
            best_pair: Optional[Tuple[str, str]] = None
            best_score = -1.0
            names = list(schema.persona_axes.keys())
            for idx, left_name in enumerate(names):
                left = schema.persona_axes[left_name]
                if left.direction is None or left.positive_anchor is None:
                    continue
                for right_name in names[idx + 1 :]:
                    right = schema.persona_axes[right_name]
                    if right.direction is None or right.positive_anchor is None:
                        continue
                    direction_sim = cosine_sim(left.direction, right.direction)
                    anchor_sim = cosine_sim(left.positive_anchor, right.positive_anchor)
                    overlap = set(self.axis_texts.get(left.name, [])) & set(self.axis_texts.get(right.name, []))
                    score = 0.55 * direction_sim + 0.35 * anchor_sim + 0.10 * min(len(overlap), 3)
                    if score > best_score:
                        best_score = score
                        best_pair = (left_name, right_name)

            if best_pair is None:
                break

            left_name, right_name = best_pair
            left = schema.persona_axes[left_name]
            right = schema.persona_axes[right_name]
            overlap = list(set(self.axis_texts.get(left.name, [])) & set(self.axis_texts.get(right.name, [])))
            should_merge, reason = narrator.judge_axis_merge(left, right, overlap)
            forced_by_budget = len(schema.persona_axes) > self.budget and best_score >= 0.82
            if not should_merge and not forced_by_budget:
                break

            canonical, alias = (left, right) if left.support_count >= right.support_count else (right, left)
            schema.merge_persona_axes(canonical.name, alias.name, note=reason or f"similarity={best_score:.3f}")
            schema.persona_axes[canonical.name].compile(axis_embedder)
            merges.append((canonical.name, alias.name))
            if len(schema.persona_axes) <= self.budget and best_score < 0.90:
                break

        self.events_since_last = 0
        return merges

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "every_events": self.every_events,
            "budget": self.budget,
            "events_since_last": self.events_since_last,
            "axis_texts": {key: list(value) for key, value in self.axis_texts.items()},
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "SchemaConsolidator":
        obj = cls(
            every_events=int(data.get("every_events", 50)),
            budget=int(data.get("budget", 24)),
        )
        obj.events_since_last = int(data.get("events_since_last", 0))
        obj.axis_texts = {str(key): [str(item) for item in value] for key, value in data.get("axis_texts", {}).items()}
        return obj


@dataclass(frozen=True)
class SoulConfig:
    dim: int = 384
    metric_rank: int = 64
    max_plots: int = 5000
    kde_reservoir: int = 4096
    subconscious_reservoir: int = 1024
    story_alpha: float = 1.0
    theme_alpha: float = 0.6
    gate_feature_dim: int = 8
    retrieval_kinds: Tuple[str, ...] = ("theme", "story", "plot")
    mode_refractory_steps: int = 4
    mode_new_threshold: float = 0.52
    encode_min_events_before_gating: int = 6
    max_recent_texts: int = 12
    profile_text: str = ""
    persona_axes_json: Optional[str] = None
    axis_merge_every_events: int = 50
    persona_axis_budget: int = 24

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "metric_rank": self.metric_rank,
            "max_plots": self.max_plots,
            "kde_reservoir": self.kde_reservoir,
            "subconscious_reservoir": self.subconscious_reservoir,
            "story_alpha": self.story_alpha,
            "theme_alpha": self.theme_alpha,
            "gate_feature_dim": self.gate_feature_dim,
            "retrieval_kinds": list(self.retrieval_kinds),
            "mode_refractory_steps": self.mode_refractory_steps,
            "mode_new_threshold": self.mode_new_threshold,
            "encode_min_events_before_gating": self.encode_min_events_before_gating,
            "max_recent_texts": self.max_recent_texts,
            "profile_text": self.profile_text,
            "persona_axes_json": self.persona_axes_json,
            "axis_merge_every_events": self.axis_merge_every_events,
            "persona_axis_budget": self.persona_axis_budget,
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "SoulConfig":
        return cls(
            dim=int(data.get("dim", 384)),
            metric_rank=int(data.get("metric_rank", 64)),
            max_plots=int(data.get("max_plots", 5000)),
            kde_reservoir=int(data.get("kde_reservoir", 4096)),
            subconscious_reservoir=int(data.get("subconscious_reservoir", 1024)),
            story_alpha=float(data.get("story_alpha", 1.0)),
            theme_alpha=float(data.get("theme_alpha", 0.6)),
            gate_feature_dim=int(data.get("gate_feature_dim", 8)),
            retrieval_kinds=tuple(data.get("retrieval_kinds", ("theme", "story", "plot"))),
            mode_refractory_steps=int(data.get("mode_refractory_steps", 4)),
            mode_new_threshold=float(data.get("mode_new_threshold", 0.52)),
            encode_min_events_before_gating=int(data.get("encode_min_events_before_gating", 6)),
            max_recent_texts=int(data.get("max_recent_texts", 12)),
            profile_text=str(data.get("profile_text", "")),
            persona_axes_json=data.get("persona_axes_json"),
            axis_merge_every_events=int(data.get("axis_merge_every_events", 50)),
            persona_axis_budget=int(data.get("persona_axis_budget", 24)),
        )


class AuroraSoul:
    def __init__(
        self,
        cfg: SoulConfig = SoulConfig(),
        *,
        seed: int = 0,
        event_embedder: Optional[EmbeddingProvider] = None,
        axis_embedder: Optional[EmbeddingProvider] = None,
        meaning_provider: Optional[MeaningProvider] = None,
        narrator: Optional[NarrativeProvider] = None,
    ) -> None:
        self.cfg = cfg
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        self.event_embedder = event_embedder or LocalSemanticEmbedding(dim=cfg.dim, seed=seed)
        self.axis_embedder = axis_embedder or self.event_embedder
        self.meaning_provider = meaning_provider or HeuristicMeaningProvider()
        self.narrator = narrator or CombinatorialNarrativeProvider()
        self.fact_extractor = FactExtractor()

        persona_axes = self._load_persona_axes()
        self.schema = schema_from_profile(
            axis_embedder=self.axis_embedder,
            profile_text=cfg.profile_text,
            persona_axes=persona_axes,
        )

        self.kde = OnlineKDE(dim=cfg.dim, reservoir=cfg.kde_reservoir, seed=seed)
        self.metric = LowRankMetric(dim=cfg.dim, rank=cfg.metric_rank, seed=seed)
        self.gate = ThompsonBernoulliGate(feature_dim=cfg.gate_feature_dim, seed=seed)
        self.graph = MemoryGraph()
        self.vindex = VectorIndex(dim=cfg.dim)
        self.retriever = FieldRetriever(metric=self.metric, vindex=self.vindex, graph=self.graph)
        self.story_model = StoryModel(metric=self.metric)
        self.theme_model = ThemeModel(metric=self.metric)
        self.crp_story = CRPAssigner(alpha=cfg.story_alpha, seed=seed)
        self.crp_theme = CRPAssigner(alpha=cfg.theme_alpha, seed=seed + 1)
        self.subconscious = SubconsciousField(reservoir=cfg.subconscious_reservoir, seed=seed)
        self.consolidator = SchemaConsolidator(
            every_events=cfg.axis_merge_every_events,
            budget=cfg.persona_axis_budget,
        )

        self.plots: Dict[str, Plot] = {}
        self.stories: Dict[str, StoryArc] = {}
        self.themes: Dict[str, Theme] = {}
        self.modes: Dict[str, IdentityMode] = {}
        self.recent_texts: List[str] = []
        self.step = 0
        self._recent_encoded_plot_ids: List[str] = []

        self.wake_axis_stats: Dict[str, Dict[str, float]] = {}
        for axis_name in self.schema.ordered_axis_names():
            self.wake_axis_stats[axis_name] = {"pos": 1.0, "neg": 1.0}

        self.identity = self._make_initial_identity()
        self._init_origin_mode()

    def _load_persona_axes(self) -> List[Dict[str, Any]]:
        if self.cfg.persona_axes_json:
            try:
                payload = json.loads(self.cfg.persona_axes_json)
                if isinstance(payload, list):
                    return payload
            except Exception:
                pass
        extracted = self.meaning_provider.extract_persona_axes(self.cfg.profile_text)
        if extracted:
            return extracted
        return heuristic_persona_axes(self.cfg.profile_text)

    def _make_initial_identity(self) -> IdentityState:
        axis_state = {name: 0.0 for name in self.schema.ordered_axis_names()}
        for name in self.schema.persona_axes.keys():
            axis_state[name] = 0.18
        axis_state["coherence"] = 0.15
        axis_state["regulation"] = 0.10
        self_vector = self._compose_self_vector(axis_state)
        state = IdentityState(
            self_vector=self_vector,
            axis_state=axis_state,
            intuition_axes={name: 0.0 for name in axis_state},
            current_mode_label="origin",
        )
        state.narrative_log.append("Initialized v4 generative soul.")
        return state

    def _compose_self_vector(self, axis_state: Dict[str, float]) -> np.ndarray:
        accumulator = np.zeros(self.cfg.dim, dtype=np.float32)
        for axis_name, value in axis_state.items():
            axis = self.schema.all_axes().get(axis_name)
            if axis is None or axis.direction is None:
                continue
            accumulator += float(value) * axis.direction
        if np.linalg.norm(accumulator) < 1e-6:
            accumulator += self.axis_embedder.embed(self.cfg.profile_text or "origin self")
        return l2_normalize(accumulator)

    def _axis_names(self) -> List[str]:
        names = self.schema.ordered_axis_names()
        for name in names:
            self.identity.axis_state.setdefault(name, 0.0)
            self.identity.intuition_axes.setdefault(name, 0.0)
            self.wake_axis_stats.setdefault(name, {"pos": 1.0, "neg": 1.0})
        return names

    def _init_origin_mode(self) -> None:
        mode_id = stable_uuid("mode")
        axis_proto = {key: float(value) for key, value in self.identity.axis_state.items()}
        label = self.narrator.label_mode(axis_proto, self.schema, support=1)
        mode = IdentityMode(
            id=mode_id,
            label=label,
            prototype=self.identity.self_vector.copy(),
            axis_prototype=axis_proto,
        )
        self.modes[mode_id] = mode
        self.identity.current_mode_id = mode_id
        self.identity.current_mode_label = label
        self.identity.narrative_log.append(f"Initialized mode: {label}")

    def _plot_vector_for_index(self, plot: Plot) -> np.ndarray:
        implication = np.zeros(self.cfg.dim, dtype=np.float32)
        for axis_name, value in plot.frame.axis_evidence.items():
            axis = self.schema.all_axes().get(axis_name)
            if axis is None or axis.direction is None:
                continue
            implication += float(value) * axis.direction
        if np.linalg.norm(implication) < 1e-6:
            return plot.embedding
        return l2_normalize(0.72 * plot.embedding + 0.28 * implication)

    def _story_vector_for_index(self, story: StoryArc) -> np.ndarray:
        if story.centroid is None:
            return np.zeros(self.cfg.dim, dtype=np.float32)
        return story.centroid

    def _theme_vector_for_index(self, theme: Theme) -> np.ndarray:
        if theme.prototype is None:
            return np.zeros(self.cfg.dim, dtype=np.float32)
        return theme.prototype

    def _redundancy(self, emb: np.ndarray) -> float:
        hits = self.vindex.search(emb, k=8, kind="plot")
        return max((score for _, score in hits), default=0.0)

    def _goal_relevance(self, emb: np.ndarray, context_emb: Optional[np.ndarray]) -> float:
        if context_emb is None:
            return 0.0
        return float(np.dot(l2_normalize(emb), l2_normalize(context_emb)))

    def _pred_error(self, emb: np.ndarray) -> float:
        best_story: Optional[StoryArc] = None
        best_sim = -1.0
        for story in self.stories.values():
            if story.centroid is None:
                continue
            sim = self.metric.sim(emb, story.centroid)
            if sim > best_sim:
                best_story = story
                best_sim = sim
        if best_story is None:
            return 1.0
        return 1.0 - best_sim

    def _compute_dissonance(self, frame: EventFrame, embedding: np.ndarray) -> DissonanceReport:
        axis_conflicts: Dict[str, float] = {}
        axis_alignments: Dict[str, float] = {}
        for axis_name in self._axis_names():
            current = self.identity.axis_state.get(axis_name, 0.0)
            evidence = frame.axis_evidence.get(axis_name, 0.0)
            axis_conflicts[axis_name] = max(0.0, -current * evidence)
            axis_alignments[axis_name] = max(0.0, current * evidence)

        semantic_conflict = max(0.0, -float(np.dot(self.identity.self_vector, embedding))) * frame.self_relevance
        affective_load = (
            0.30 * frame.arousal
            + 0.20 * max(0.0, -frame.valence)
            + 0.18 * frame.threat
            + 0.12 * frame.shame
            + 0.10 * frame.control
            + 0.10 * frame.abandonment
        )
        narrative_incongruity = mean_abs(axis_conflicts.values()) * (0.55 + 0.45 * frame.self_relevance)
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
        base = (
            0.42
            + 0.14 * self.identity.rigidity()
            - 0.10 * self.identity.plasticity()
            + 0.03 * min(self.identity.repair_count, 6)
        )
        return max(0.26, base)

    def _top_conflict_axes(self, dissonance: DissonanceReport, topn: int = 4) -> List[str]:
        items = sorted(dissonance.axis_conflicts.items(), key=lambda item: item[1], reverse=True)
        return [name for name, value in items[:topn] if value > 0.03]

    def _reality_fit_for_axis(self, axis_name: str, sign: float) -> float:
        stats = self.wake_axis_stats.get(axis_name, {"pos": 1.0, "neg": 1.0})
        pos = stats["pos"]
        neg = stats["neg"]
        if sign >= 0:
            return pos / (pos + neg + 1e-12)
        return neg / (pos + neg + 1e-12)

    def _candidate_repairs(self, plot: Plot, dissonance: DissonanceReport) -> List[RepairCandidate]:
        state = self.identity
        salient_axes = self._top_conflict_axes(dissonance)
        if not salient_axes:
            fallback_scores = {
                name: abs(plot.frame.axis_evidence.get(name, 0.0))
                for name in self._axis_names()
            }
            salient_axes = [name for name, value in sorted(fallback_scores.items(), key=lambda item: item[1], reverse=True)[:4] if value > 0.08]
            if not salient_axes:
                return []

        dream_support = 0.0
        for axis_name in self._axis_names():
            dream_support += state.intuition_axes.get(axis_name, 0.0) * plot.frame.axis_evidence.get(axis_name, 0.0)
        dream_support /= max(len(self._axis_names()), 1)

        repeated_conflict = float(np.mean([
            self._reality_fit_for_axis(axis_name, plot.frame.axis_evidence.get(axis_name, 0.0))
            for axis_name in salient_axes
        ]))

        def shift_axes(mode: str) -> Tuple[Dict[str, float], np.ndarray]:
            axes = {key: float(value) for key, value in state.axis_state.items()}
            for axis_name in salient_axes:
                evidence = plot.frame.axis_evidence.get(axis_name, 0.0)
                current = axes.get(axis_name, 0.0)
                if mode == "preserve":
                    if current * evidence < 0:
                        axes[axis_name] = clamp(current + 0.18 * np.sign(current if abs(current) > 0.02 else -evidence))
                elif mode == "reframe":
                    axes[axis_name] = clamp(current + 0.10 * evidence * (0.5 + state.plasticity()))
                elif mode == "revise":
                    axes[axis_name] = clamp(current + 0.22 * evidence * (0.45 + repeated_conflict))
                elif mode == "differentiate":
                    axes[axis_name] = clamp(current + 0.10 * evidence * 0.5)
                else:
                    axes[axis_name] = clamp(current + 0.16 * evidence * (0.35 + repeated_conflict + max(0.0, dream_support)))

            threat_load = plot.frame.threat + 0.7 * plot.frame.control + 0.6 * plot.frame.abandonment
            if mode == "preserve":
                axes["vigilance"] = clamp(axes.get("vigilance", 0.0) + 0.18 * threat_load)
                axes["exploration"] = clamp(axes.get("exploration", 0.0) - 0.12 * threat_load)
                axes["regulation"] = clamp(axes.get("regulation", 0.0) - 0.04 * plot.frame.arousal)
            elif mode == "reframe":
                axes["coherence"] = clamp(axes.get("coherence", 0.0) + 0.10)
                axes["regulation"] = clamp(axes.get("regulation", 0.0) + 0.06)
            elif mode == "revise":
                axes["coherence"] = clamp(axes.get("coherence", 0.0) + 0.14)
                axes["exploration"] = clamp(axes.get("exploration", 0.0) + 0.08 * repeated_conflict)
            elif mode == "differentiate":
                axes["agency"] = clamp(axes.get("agency", 0.0) + 0.22 * max(plot.frame.agency_signal, threat_load))
                axes["affiliation"] = clamp(axes.get("affiliation", 0.0) - 0.10 * threat_load)
                axes["vigilance"] = clamp(axes.get("vigilance", 0.0) + 0.12 * threat_load)
                axes["coherence"] = clamp(axes.get("coherence", 0.0) + 0.06)
            else:
                axes["coherence"] = clamp(axes.get("coherence", 0.0) + 0.16)
                axes["regulation"] = clamp(axes.get("regulation", 0.0) + 0.12)
                axes["exploration"] = clamp(axes.get("exploration", 0.0) + 0.08 * max(plot.frame.care, repeated_conflict))
                axes["agency"] = clamp(axes.get("agency", 0.0) + 0.06 * plot.frame.agency_signal)
                axes["affiliation"] = clamp(axes.get("affiliation", 0.0) + 0.08 * plot.frame.care - 0.04 * threat_load)
            return axes, self._compose_self_vector(axes)

        def score(
            mode: str,
            axes: Dict[str, float],
            new_vector: np.ndarray,
            active_after: float,
            repressed_after: float,
        ) -> RepairCandidate:
            drift = 1.0 - float(np.dot(state.self_vector, new_vector))
            coherence_gain = max(0.0, axes.get("coherence", 0.0) - state.axis_state.get("coherence", 0.0))
            current_pressure = state.narrative_pressure()
            new_contradiction = moving_average(state.contradiction_ema, dissonance.total, 0.20)
            projected_pressure = active_after + 0.85 * repressed_after + 1.20 * new_contradiction
            pressure_relief = max(0.0, current_pressure - projected_pressure)
            reality = float(np.mean([
                self._reality_fit_for_axis(axis_name, axes.get(axis_name, 0.0))
                for axis_name in salient_axes
            ]))
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
                utility += 0.14 * (plot.frame.threat + plot.frame.control + plot.frame.agency_signal)
            else:
                utility += 0.12 * state.plasticity() + 0.12 * max(0.0, dream_support) + 0.06 * plot.frame.care

            explanation = self.narrator.compose_repair(
                mode,
                state,
                IdentityState(
                    self_vector=new_vector,
                    axis_state=axes,
                    intuition_axes=dict(state.intuition_axes),
                    active_energy=active_after,
                    repressed_energy=repressed_after,
                    contradiction_ema=state.contradiction_ema,
                    current_mode_id=state.current_mode_id,
                    current_mode_label=state.current_mode_label,
                ),
                dissonance.total,
                salient_axes,
                plot.text,
                self.schema,
            )
            return RepairCandidate(
                mode=mode,
                new_vector=new_vector,
                new_axes=axes,
                new_active=max(0.0, active_after),
                new_repressed=max(0.0, repressed_after),
                coherence_gain=coherence_gain,
                identity_drift=drift,
                pressure_relief=pressure_relief,
                reality_fit=reality,
                dream_support=dream_support,
                utility=utility,
                explanation=explanation,
            )

        candidates: List[RepairCandidate] = []
        for mode, active_after, repressed_after in [
            ("preserve", state.active_energy - 0.28 * dissonance.total, state.repressed_energy + 0.52 * dissonance.total),
            ("reframe", state.active_energy - 0.48 * dissonance.total, state.repressed_energy + 0.16 * dissonance.total),
            ("revise", state.active_energy - 0.72 * dissonance.total, state.repressed_energy + 0.06 * dissonance.total),
            ("differentiate", state.active_energy - 0.60 * dissonance.total, state.repressed_energy + 0.12 * dissonance.total),
            ("integrate", state.active_energy - 0.82 * dissonance.total, max(0.0, state.repressed_energy - 0.10 * dissonance.total)),
        ]:
            axes, new_vector = shift_axes(mode)
            candidates.append(score(mode, axes, new_vector, active_after, repressed_after))
        return candidates

    def _apply_candidate_repair(self, candidate: RepairCandidate, plot: Plot) -> None:
        self.identity.self_vector = candidate.new_vector
        self.identity.axis_state = copy.deepcopy(candidate.new_axes)
        self.identity.active_energy = candidate.new_active
        self.identity.repressed_energy = candidate.new_repressed
        self.identity.repair_count += 1
        self.identity.narrative_log.append(candidate.explanation)
        repair_plot = self.ingest(
            candidate.explanation,
            actors=("self",),
            source="repair",
            confidence=0.72,
            evidence_weight=0.35,
        )
        if repair_plot.id in self.graph.g and plot.id in self.graph.g:
            self.graph.ensure_edge(repair_plot.id, plot.id, "rationalizes")

    def _maybe_reconstruct(self, plot: Plot, dissonance: DissonanceReport) -> None:
        pressure = self.identity.narrative_pressure()
        threshold = self._dynamic_repair_threshold()
        activation = pressure + 0.65 * plot.tension + 0.75 * dissonance.total
        p_repair = sigmoid(activation - threshold)
        if activation <= threshold and self.rng.random() >= p_repair:
            self.identity.repressed_energy += 0.22 * dissonance.total
            return
        candidates = self._candidate_repairs(plot, dissonance)
        if not candidates:
            self.identity.repressed_energy += 0.15 * dissonance.total
            return
        probs = softmax([candidate.utility for candidate in candidates])
        idx = int(self.rng.choice(np.arange(len(candidates)), p=np.asarray(probs, dtype=np.float64)))
        self._apply_candidate_repair(candidates[idx], plot)

    def _current_axis_signature(self) -> Dict[str, float]:
        return {key: float(value) for key, value in self.identity.axis_state.items()}

    def _maybe_mode_transition(self, cause_plot: Plot) -> Optional[IdentityMode]:
        if self.step - self.identity.last_mode_step < self.cfg.mode_refractory_steps:
            return None

        pressure = self.identity.narrative_pressure()
        current_mode = self.modes.get(self.identity.current_mode_id) if self.identity.current_mode_id else None
        current_score = current_mode.score(self.identity.self_vector, self.identity.axis_state) if current_mode else 0.0

        best_mode = current_mode
        best_score = current_score
        for mode in self.modes.values():
            score = mode.score(self.identity.self_vector, self.identity.axis_state)
            if score > best_score:
                best_mode = mode
                best_score = score

        drift_from_current = 1.0 - current_score
        create_new = (
            pressure > 0.95
            and drift_from_current > self.cfg.mode_new_threshold
            and (best_mode is current_mode or best_score < 0.76)
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
        if current_mode is not None and best_mode.id == current_mode.id:
            current_mode.prototype = l2_normalize(0.92 * current_mode.prototype + 0.08 * self.identity.self_vector)
            for key, value in self.identity.axis_state.items():
                current_mode.axis_prototype[key] = moving_average(current_mode.axis_prototype.get(key, 0.0), value, 0.08)
            current_mode.support += 1
            current_mode.updated_ts = now_ts()
            return None

        threshold = best_mode.barrier + best_mode.hysteresis * (1.0 + 0.35 * self.identity.mode_change_count)
        if best_score - current_score <= threshold:
            return None

        previous_label = self.identity.current_mode_label
        self.identity.current_mode_id = best_mode.id
        self.identity.current_mode_label = best_mode.label
        self.identity.mode_change_count += 1
        self.identity.last_mode_step = self.step
        self.identity.last_mode_change_ts = now_ts()
        self.identity.narrative_log.append(f"Mode transition: {previous_label} -> {best_mode.label}")

        mode_plot = self.ingest(
            f"[mode] She moves from '{previous_label}' into '{best_mode.label}'.",
            actors=("self",),
            source="mode",
            confidence=0.76,
            evidence_weight=0.40,
        )
        if mode_plot.id in self.graph.g and cause_plot.id in self.graph.g:
            self.graph.ensure_edge(mode_plot.id, cause_plot.id, "triggered_by")
        return best_mode

    def _assign_story(self, plot: Plot) -> StoryArc:
        logps: Dict[str, float] = {}
        for story_id, story in self.stories.items():
            log_prior = math.log(len(story.plot_ids) + 1.0)
            logps[story_id] = log_prior + self.story_model.loglik(plot, story)
        chosen_story_id, _ = self.crp_story.sample(logps)
        if chosen_story_id is None:
            chosen_story_id = stable_uuid("story")
            story = StoryArc(id=chosen_story_id, created_ts=plot.ts, updated_ts=plot.ts)
            self.stories[story.id] = story
            self.graph.add_node(story.id, "story", story)
            self.vindex.add(story.id, plot.embedding, kind="story")
        story = self.stories[chosen_story_id]
        if story.centroid is None:
            story.centroid = plot.embedding.copy()
        else:
            story._update_stats("dist", self.metric.d2(plot.embedding, story.centroid))
            story._update_stats("gap", max(0.0, plot.ts - story.updated_ts))
            story.centroid = l2_normalize(0.90 * story.centroid + 0.10 * plot.embedding)
        story.plot_ids.append(plot.id)
        story.updated_ts = plot.ts
        story.unresolved_energy += plot.contradiction
        story.source_counts[plot.source] = story.source_counts.get(plot.source, 0) + 1
        for actor in plot.actors:
            story.actor_counts[actor] = story.actor_counts.get(actor, 0) + 1
        for tag in plot.frame.tags[:8]:
            story.tag_counts[tag] = story.tag_counts.get(tag, 0) + 1
        story.tension_curve.append(plot.tension)
        plot.story_id = story.id
        self.graph.ensure_edge(plot.id, story.id, "belongs_to")
        self.graph.ensure_edge(story.id, plot.id, "contains")
        if story.id in self.vindex.ids:
            idx = self.vindex.ids.index(story.id)
            self.vindex.vecs[idx] = self._story_vector_for_index(story)
        return story

    def _assign_theme(self, story: StoryArc) -> Optional[Theme]:
        if story.centroid is None:
            return None
        logps: Dict[str, float] = {}
        for theme_id, theme in self.themes.items():
            log_prior = math.log(len(theme.story_ids) + 1.0)
            logps[theme_id] = log_prior + self.theme_model.loglik(story, theme)
        chosen_theme_id, _ = self.crp_theme.sample(logps)
        if chosen_theme_id is None:
            chosen_theme_id = stable_uuid("theme")
            label = self._label_theme(story)
            theme = Theme(
                id=chosen_theme_id,
                created_ts=now_ts(),
                updated_ts=now_ts(),
                story_ids=[],
                prototype=story.centroid.copy(),
                label=label,
                name=label,
                description=f"Emergent theme around: {label}",
            )
            self.themes[theme.id] = theme
            self.graph.add_node(theme.id, "theme", theme)
            self.vindex.add(theme.id, theme.prototype, kind="theme")
        theme = self.themes[chosen_theme_id]
        if story.id not in theme.story_ids:
            theme.story_ids.append(story.id)
        theme.updated_ts = now_ts()
        if theme.prototype is None:
            theme.prototype = story.centroid.copy()
        else:
            theme.prototype = l2_normalize(0.90 * theme.prototype + 0.10 * story.centroid)
        theme.label = theme.label or theme.name or self._label_theme(story)
        theme.name = theme.name or theme.label
        self.graph.ensure_edge(story.id, theme.id, "instantiates")
        self.graph.ensure_edge(theme.id, story.id, "grounds")
        if theme.id in self.vindex.ids:
            idx = self.vindex.ids.index(theme.id)
            self.vindex.vecs[idx] = self._theme_vector_for_index(theme)
        return theme

    def _label_theme(self, story: StoryArc) -> str:
        tags = sorted(story.tag_counts.items(), key=lambda item: item[1], reverse=True)
        if not tags:
            return "untitled_theme"
        return " / ".join([tag for tag, _ in tags[:3]])

    def _maybe_store_fragment(self, plot: Plot) -> None:
        fragment = LatentFragment(
            plot_id=plot.id,
            ts=plot.ts,
            activation=clamp01(0.30 + 0.35 * plot.frame.arousal + 0.35 * plot.frame.self_relevance),
            unresolved=clamp01(0.55 * plot.contradiction + 0.45 * plot.tension),
            embedding=plot.embedding.copy(),
            axis_evidence={key: float(value) for key, value in plot.frame.axis_evidence.items()},
            source=plot.source,
            tags=tuple(plot.frame.tags[:10]),
        )
        self.subconscious.add(fragment)

    def _store_plot(self, plot: Plot) -> None:
        self.plots[plot.id] = plot
        self.graph.add_node(plot.id, "plot", plot)
        self.vindex.add(plot.id, self._plot_vector_for_index(plot), kind="plot")
        story = self._assign_story(plot)
        theme = self._assign_theme(story)
        if theme is not None:
            plot.theme_id = theme.id
            self.graph.ensure_edge(plot.id, theme.id, "suggests_theme")
            self.graph.ensure_edge(theme.id, plot.id, "evidenced_by")
        self._maybe_store_fragment(plot)

    def _baseline_assimilation(self, frame: EventFrame) -> None:
        changed = False
        for axis_name in self._axis_names():
            evidence = frame.axis_evidence.get(axis_name, 0.0)
            if abs(evidence) < 0.06:
                continue
            axis = self.schema.all_axes().get(axis_name)
            rate = 0.015 if (axis is not None and axis.level == "persona") else 0.025
            rate *= (0.55 + 0.45 * frame.self_relevance)
            self.identity.axis_state[axis_name] = moving_average(
                self.identity.axis_state.get(axis_name, 0.0),
                evidence,
                rate,
            )
            changed = True
        self.identity.axis_state["regulation"] = clamp(
            self.identity.axis_state.get("regulation", 0.0) + 0.03 * frame.care - 0.04 * frame.arousal - 0.03 * frame.threat
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
        for axis_name in self._axis_names():
            value = frame.axis_evidence.get(axis_name, 0.0)
            if value >= 0:
                self.wake_axis_stats[axis_name]["pos"] += abs(value) + 1e-3
            else:
                self.wake_axis_stats[axis_name]["neg"] += abs(value) + 1e-3

    def _dream_update_intuition(self, frame: EventFrame, resonance: float) -> None:
        gain = 0.10 + 0.12 * sigmoid(resonance)
        for axis_name in self._axis_names():
            evidence = frame.axis_evidence.get(axis_name, 0.0)
            self.identity.intuition_axes[axis_name] = moving_average(
                self.identity.intuition_axes.get(axis_name, 0.0),
                gain * evidence,
                0.14,
            )

    def _promote_dream_intuitions(self) -> None:
        for axis_name in self._axis_names():
            bias = self.identity.intuition_axes.get(axis_name, 0.0)
            if abs(bias) < 1e-6:
                continue
            corroboration = self._reality_fit_for_axis(axis_name, bias)
            strength = abs(bias) * corroboration
            if self.rng.random() < sigmoid(3.0 * (strength - 0.35)):
                self.identity.axis_state[axis_name] = clamp(self.identity.axis_state.get(axis_name, 0.0) + 0.05 * bias)
        self.identity.self_vector = self._compose_self_vector(self.identity.axis_state)

    def _encode_features(self, plot: Plot) -> np.ndarray:
        return np.asarray(
            [
                clamp01(plot.surprise / 3.0),
                clamp01(plot.pred_error),
                clamp01(1.0 - plot.redundancy),
                clamp01(plot.goal_relevance),
                clamp01(plot.contradiction / 1.5),
                clamp01(plot.frame.arousal),
                clamp01(plot.frame.novelty),
                1.0 if plot.source == "wake" else 0.5,
            ],
            dtype=np.float32,
        )

    def _maybe_expand_schema(self, plot: Plot) -> bool:
        if plot.source != "wake" or plot.tension < 0.75:
            return False
        existing = set(self.schema.all_axes().keys())
        for tag in plot.frame.tags:
            if ":" in tag or tag.startswith("echo:") or tag in existing:
                continue
            axis = AxisSpec(
                name=tag[:24],
                positive_pole=tag,
                negative_pole=f"not_{tag}",
                description=f"Discovered from repeated unresolved motif: {tag}",
                level="persona",
            )
            if axis.name in self.schema.persona_axes:
                continue
            self.schema.add_persona_axis(axis, self.axis_embedder)
            self.identity.axis_state.setdefault(axis.name, 0.0)
            self.identity.intuition_axes.setdefault(axis.name, 0.0)
            self.wake_axis_stats.setdefault(axis.name, {"pos": 1.0, "neg": 1.0})
            self.identity.narrative_log.append(f"New axis emerged: {axis.name} ({axis.positive_pole} ↔ {axis.negative_pole})")
            return True
        return False

    def _pressure_manage(self) -> None:
        if len(self.plots) <= self.cfg.max_plots:
            return
        victims = sorted(self.plots.values(), key=lambda plot: plot.mass())[: max(1, len(self.plots) - self.cfg.max_plots)]
        for victim in victims:
            victim.status = "archived"
            self.vindex.remove(victim.id)

    def ingest(
        self,
        interaction_text: str,
        *,
        actors: Optional[Sequence[str]] = None,
        context_text: Optional[str] = None,
        source: Literal["wake", "dream", "repair", "mode"] = "wake",
        confidence: float = 1.0,
        evidence_weight: float = 1.0,
        ts: Optional[float] = None,
        plot_id: Optional[str] = None,
    ) -> Plot:
        self.step += 1
        event_ts = ts or now_ts()
        embedding = self.event_embedder.embed(interaction_text)
        frame = self.meaning_provider.extract(
            interaction_text,
            embedding,
            self.schema,
            recent_tags=self.recent_texts,
        )
        plot = Plot(
            id=plot_id or str(uuid.uuid4()),
            ts=event_ts,
            text=interaction_text,
            actors=tuple(actors) if actors else ("user", "agent"),
            embedding=embedding,
            frame=frame,
            source=source,
            confidence=confidence,
            evidence_weight=evidence_weight,
            fact_keys=[fact.fact_text for fact in self.fact_extractor.extract(interaction_text)],
        )

        context_emb = self.event_embedder.embed(context_text) if context_text else None
        plot.surprise = float(self.kde.surprise(embedding))
        plot.pred_error = float(self._pred_error(embedding))
        plot.redundancy = float(self._redundancy(embedding))
        plot.goal_relevance = float(self._goal_relevance(embedding, context_emb))

        dissonance = self._compute_dissonance(frame, embedding)
        plot.contradiction = float(dissonance.total)
        plot.tension = float(
            (
                0.30 * plot.surprise
                + 0.20 * plot.pred_error
                + 0.20 * plot.contradiction
                + 0.15 * frame.arousal
                + 0.10 * frame.self_relevance
                + 0.05 * max(0.0, 1.0 - plot.redundancy)
            )
            * evidence_weight
        )

        encode = source in {"dream", "repair", "mode"} or len(self.plots) < self.cfg.encode_min_events_before_gating
        if not encode:
            encode = self.gate.decide(self._encode_features(plot))
        if encode:
            self._store_plot(plot)
            self.gate.update(self._encode_features(plot), reward=1.0)
            self._recent_encoded_plot_ids.append(plot.id)
            self._recent_encoded_plot_ids = self._recent_encoded_plot_ids[-250:]
        else:
            self.gate.update(self._encode_features(plot), reward=0.0)

        energy_scale = {"wake": 1.0, "dream": 0.35, "repair": 0.20, "mode": 0.15}[source]
        self.identity.active_energy += energy_scale * (
            0.38 * dissonance.total + 0.26 * frame.arousal + 0.16 * max(0.0, -frame.valence) + 0.10 * frame.threat
        )
        self.identity.contradiction_ema = moving_average(self.identity.contradiction_ema, dissonance.total, 0.24)

        self.subconscious.add(
            LatentFragment(
                plot_id=plot.id,
                ts=plot.ts,
                activation=clamp01(0.30 + 0.35 * frame.arousal + 0.35 * frame.self_relevance),
                unresolved=clamp01(0.55 * plot.contradiction + 0.45 * plot.tension),
                embedding=plot.embedding.copy(),
                axis_evidence={key: float(value) for key, value in frame.axis_evidence.items()},
                source=source,
                tags=tuple(frame.tags[:10]),
            )
        )

        if source == "wake":
            self._update_wake_axis_stats(frame)
            self._maybe_reconstruct(plot, dissonance)
            self._baseline_assimilation(frame)
            added_axis = self._maybe_expand_schema(plot)
            self.consolidator.observe(plot)
            if self.consolidator.should_run(added_axis, len(self.schema.persona_axes)):
                merges = self.consolidator.maybe_consolidate(
                    schema=self.schema,
                    axis_embedder=self.axis_embedder,
                    narrator=self.narrator,
                )
                for canonical, alias in merges:
                    self.identity.narrative_log.append(f"Axis merged: {alias} -> {canonical}")
                self._axis_names()
            self._promote_dream_intuitions()
            self._maybe_mode_transition(plot)

        self.recent_texts.append(interaction_text)
        self.recent_texts = self.recent_texts[-self.cfg.max_recent_texts :]
        self._pressure_manage()
        return plot

    def dream(self, n: int = 2) -> List[Plot]:
        dreams: List[Plot] = []
        for _ in range(n):
            fragments = self.subconscious.sample(n=int(self.rng.integers(1, 4)))
            if not fragments:
                continue
            text, frame, resonance = self.subconscious.synthesize(fragments, self.identity, self.schema, self.narrator)
            plot = self.ingest(
                text,
                actors=("self",),
                source="dream",
                confidence=0.34,
                evidence_weight=0.20,
            )
            self._dream_update_intuition(frame, resonance)
            self.identity.dream_count += 1
            dreams.append(plot)
        if dreams:
            self._promote_dream_intuitions()
        return dreams

    def evolve(self, dreams: int = 2) -> List[Plot]:
        return self.dream(n=dreams)

    def query(self, text: str, k: int = 8):
        trace = self.retriever.retrieve(
            query_text=text,
            embedder=self.event_embedder,
            state=self.identity,
            kinds=self.cfg.retrieval_kinds,
            k=k,
        )
        for node_id, _, kind in trace.ranked:
            if kind == "plot" and node_id in self.plots:
                self.plots[node_id].access_count += 1
                self.plots[node_id].last_access_ts = now_ts()
            elif kind == "story" and node_id in self.stories:
                self.stories[node_id].reference_count += 1
        return trace

    def feedback_retrieval(self, query_text: str, chosen_id: str, success: bool) -> None:
        if chosen_id not in self.graph.g:
            return
        query_vec = self.event_embedder.embed(query_text)
        chosen_vec = self.retriever._payload_vec(self.graph.payload(chosen_id))
        if chosen_vec is None:
            return
        coarse = self.vindex.search(query_vec, k=6)
        negative_id = next((node_id for node_id, _ in coarse if node_id != chosen_id), None)
        if negative_id is not None:
            negative_vec = self.retriever._payload_vec(self.graph.payload(negative_id))
            if negative_vec is not None:
                if success:
                    self.metric.update_triplet(query_vec, chosen_vec, negative_vec)
                else:
                    self.metric.update_triplet(query_vec, negative_vec, chosen_vec)

        if self.graph.kind(chosen_id) == "theme" and chosen_id in self.themes:
            self.themes[chosen_id].update_evidence(success)
        for neighbor in list(self.graph.g.predecessors(chosen_id)) + list(self.graph.g.successors(chosen_id)):
            if self.graph.g.has_edge(neighbor, chosen_id):
                self.graph.edge_belief(neighbor, chosen_id).update(success)
            if self.graph.g.has_edge(chosen_id, neighbor):
                self.graph.edge_belief(chosen_id, neighbor).update(success)

    def intuition_keywords(self, limit: int = 2) -> List[str]:
        items = sorted(self.identity.intuition_axes.items(), key=lambda item: abs(item[1]), reverse=True)
        phrases: List[str] = []
        for axis_name, value in items[:limit]:
            axis = self.schema.all_axes().get(axis_name)
            if axis is None or abs(value) < 0.05:
                continue
            pole = axis.positive_pole if value >= 0 else axis.negative_pole
            phrases.append(f"{pole}({value:+.2f})")
        return phrases

    def snapshot_identity(self) -> IdentitySnapshot:
        ordered = self._axis_names()
        return IdentitySnapshot(
            current_mode=self.identity.current_mode_label,
            axis_state={name: round(float(self.identity.axis_state.get(name, 0.0)), 4) for name in ordered},
            intuition_axes={name: round(float(self.identity.intuition_axes.get(name, 0.0)), 4) for name in ordered},
            persona_axes={
                name: {
                    "positive_pole": axis.positive_pole,
                    "negative_pole": axis.negative_pole,
                    "description": axis.description,
                    "support_count": axis.support_count,
                    "aliases": list(axis.aliases),
                }
                for name, axis in self.schema.persona_axes.items()
            },
            axis_aliases=dict(self.schema.axis_aliases),
            modes={
                mode_id: {
                    "label": mode.label,
                    "support": mode.support,
                    "barrier": round(mode.barrier, 4),
                    "hysteresis": round(mode.hysteresis, 4),
                }
                for mode_id, mode in self.modes.items()
            },
            active_energy=round(self.identity.active_energy, 4),
            repressed_energy=round(self.identity.repressed_energy, 4),
            contradiction_ema=round(self.identity.contradiction_ema, 4),
            plasticity=round(self.identity.plasticity(), 4),
            rigidity=round(self.identity.rigidity(), 4),
            repair_count=self.identity.repair_count,
            dream_count=self.identity.dream_count,
            mode_change_count=self.identity.mode_change_count,
            narrative_tail=self.identity.narrative_log[-8:],
        )

    def narrative_summary(self) -> NarrativeSummary:
        return self.narrator.compose_summary(self.identity, self.schema, self.recent_texts)

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SOUL_MEMORY_STATE_VERSION,
            "cfg": self.cfg.to_state_dict(),
            "kde": self.kde.to_state_dict(),
            "metric": self.metric.to_state_dict(),
            "gate": self.gate.to_state_dict(),
            "graph": self.graph.to_state_dict(),
            "vindex": self.vindex.to_state_dict(),
            "plots": {plot_id: plot.to_state_dict() for plot_id, plot in self.plots.items()},
            "stories": {story_id: story.to_state_dict() for story_id, story in self.stories.items()},
            "themes": {theme_id: theme.to_state_dict() for theme_id, theme in self.themes.items()},
            "crp_story": self.crp_story.to_state_dict(),
            "crp_theme": self.crp_theme.to_state_dict(),
            "subconscious": self.subconscious.to_state_dict(),
            "schema": self.schema.to_state_dict(),
            "consolidator": self.consolidator.to_state_dict(),
            "identity": self.identity.to_state_dict(),
            "modes": {mode_id: mode.to_state_dict() for mode_id, mode in self.modes.items()},
            "recent_texts": list(self.recent_texts),
            "step": int(self.step),
            "wake_axis_stats": copy.deepcopy(self.wake_axis_stats),
        }

    @classmethod
    def from_state_dict(
        cls,
        data: Dict[str, Any],
        *,
        event_embedder: Optional[EmbeddingProvider] = None,
        axis_embedder: Optional[EmbeddingProvider] = None,
        meaning_provider: Optional[MeaningProvider] = None,
        narrator: Optional[NarrativeProvider] = None,
    ) -> "AuroraSoul":
        if data.get("schema_version") != SOUL_MEMORY_STATE_VERSION:
            raise ValueError("Snapshot schema mismatch: expected Aurora Soul v4")
        cfg = SoulConfig.from_state_dict(data["cfg"])
        obj = cls(
            cfg=cfg,
            seed=0,
            event_embedder=event_embedder,
            axis_embedder=axis_embedder,
            meaning_provider=meaning_provider,
            narrator=narrator,
        )
        obj.kde = OnlineKDE.from_state_dict(data["kde"])
        obj.metric = LowRankMetric.from_state_dict(data["metric"])
        obj.gate = ThompsonBernoulliGate.from_state_dict(data["gate"])
        obj.vindex = VectorIndex.from_state_dict(data["vindex"])
        obj.plots = {plot_id: Plot.from_state_dict(item) for plot_id, item in data.get("plots", {}).items()}
        obj.stories = {story_id: StoryArc.from_state_dict(item) for story_id, item in data.get("stories", {}).items()}
        obj.themes = {theme_id: Theme.from_state_dict(item) for theme_id, item in data.get("themes", {}).items()}
        obj.graph = MemoryGraph()
        for plot_id, plot in obj.plots.items():
            obj.graph.add_node(plot_id, "plot", plot)
        for story_id, story in obj.stories.items():
            obj.graph.add_node(story_id, "story", story)
        for theme_id, theme in obj.themes.items():
            obj.graph.add_node(theme_id, "theme", theme)
        obj.graph.restore_edges(data["graph"])
        obj.retriever = FieldRetriever(metric=obj.metric, vindex=obj.vindex, graph=obj.graph)
        obj.crp_story = CRPAssigner.from_state_dict(data["crp_story"])
        obj.crp_theme = CRPAssigner.from_state_dict(data["crp_theme"])
        obj.subconscious = SubconsciousField.from_state_dict(data["subconscious"])
        obj.schema = PsychologicalSchema.from_state_dict(data["schema"])
        obj.schema.compile(obj.axis_embedder)
        obj.consolidator = SchemaConsolidator.from_state_dict(data.get("consolidator", {}))
        obj.identity = IdentityState.from_state_dict(data["identity"])
        obj.modes = {mode_id: IdentityMode.from_state_dict(item) for mode_id, item in data.get("modes", {}).items()}
        obj.recent_texts = [str(item) for item in data.get("recent_texts", [])]
        obj.step = int(data.get("step", 0))
        obj.wake_axis_stats = copy.deepcopy(data.get("wake_axis_stats", {}))
        obj.story_model = StoryModel(metric=obj.metric)
        obj.theme_model = ThemeModel(metric=obj.metric)
        return obj
