from __future__ import annotations

import copy
import math
import random
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from aurora.core.soul_memory.extractors import HeuristicMeaningExtractor, MeaningExtractor
from aurora.core.soul_memory.models import (
    BELIEF_ORDER,
    TRAIT_ORDER,
    CandidateRepair,
    DissonanceReport,
    EventFrame,
    IdentitySnapshot,
    IdentityState,
    LatentFragment,
    NarrativeSummary,
    PhaseProfile,
    Plot,
    StoryArc,
    Theme,
    clamp,
    mean_abs,
    signed01,
    summarize_identity,
)
from aurora.core.soul_memory.retrieval import (
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
from aurora.integrations.embeddings.base import EmbeddingProvider
from aurora.integrations.embeddings.hash import HashEmbedding
from aurora.utils.math_utils import l2_normalize, sigmoid
from aurora.utils.time_utils import now_ts

SOUL_MEMORY_STATE_VERSION = "aurora-soul-memory-v3"


class SubconsciousField:
    def __init__(self, reservoir: int = 1024, seed: int = 0):
        self.reservoir = reservoir
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self.fragments: List[LatentFragment] = []

    def deposit(self, plot: Plot, unresolved: float) -> None:
        fragment = LatentFragment(
            plot_id=plot.id,
            ts=plot.ts,
            activation=max(0.05, plot.tension),
            unresolved=max(0.0, unresolved),
            trait_vector=plot.frame.trait_vector(),
            source=plot.source,
            tags=plot.frame.tags,
        )
        if len(self.fragments) < self.reservoir:
            self.fragments.append(fragment)
            return
        idx = int(self.rng.integers(0, len(self.fragments)))
        old_score = self.fragments[idx].activation + 0.5 * self.fragments[idx].unresolved
        new_score = fragment.activation + 0.5 * fragment.unresolved
        if new_score >= old_score:
            self.fragments[idx] = fragment

    def sample(self, n: int = 2) -> List[LatentFragment]:
        if not self.fragments:
            return []
        now = now_ts()
        weights = []
        for fragment in self.fragments:
            age = max(1.0, now - fragment.ts)
            recency = 1.0 / math.log1p(age)
            weight = (0.4 + fragment.activation) * (0.4 + fragment.unresolved) * recency
            weights.append(max(weight, 1e-6))
        probs = np.asarray(weights, dtype=np.float64)
        probs /= probs.sum()
        size = min(n, len(self.fragments))
        idxs = self.rng.choice(np.arange(len(self.fragments)), size=size, replace=False, p=probs)
        return [self.fragments[int(idx)] for idx in idxs]

    def dream_operator(self, fragments: List[LatentFragment], state: IdentityState) -> str:
        tags = set()
        for fragment in fragments:
            tags.update(fragment.tags)
        if "betrayal" in tags or "rejection" in tags or "loss" in tags:
            if state.traits["autonomy"] < 0.60:
                return "counterfactual_boundary"
            return "guarded_projection"
        if "care" in tags and state.traits["defensiveness"] > 0.55:
            return "repair_rehearsal"
        if state.traits["coherence"] < 0.55:
            return "integration_bridge"
        return "exploratory_blend"

    def synthesize(self, fragments: List[LatentFragment], state: IdentityState) -> Tuple[str, EventFrame, float]:
        if not fragments:
            frame = EventFrame(
                trait_evidence={name: 0.0 for name in TRAIT_ORDER},
                belief_evidence={name: 0.0 for name in BELIEF_ORDER},
                valence=0.0,
                arousal=0.0,
                tags=("empty_dream",),
            )
            return "空梦。", frame, 0.0

        operator = self.dream_operator(fragments, state)
        base = np.mean([fragment.trait_vector for fragment in fragments], axis=0)
        base = np.clip(base, -1.0, 1.0)
        trait_evidence = {name: float(base[idx]) for idx, name in enumerate(TRAIT_ORDER)}
        belief_evidence = {name: 0.0 for name in BELIEF_ORDER}
        valence = -0.10
        arousal = 0.35
        tags = {"dream", operator}

        if operator == "counterfactual_boundary":
            trait_evidence["autonomy"] = clamp(trait_evidence.get("autonomy", 0.0) + 0.55, -1.0, 1.0)
            trait_evidence["assertiveness"] = clamp(trait_evidence.get("assertiveness", 0.0) + 0.60, -1.0, 1.0)
            trait_evidence["attachment"] = clamp(trait_evidence.get("attachment", 0.0) - 0.28, -1.0, 1.0)
            trait_evidence["coherence"] = clamp(trait_evidence.get("coherence", 0.0) + 0.18, -1.0, 1.0)
            belief_evidence["boundaries_allowed"] = 0.75
            belief_evidence["independence_safe"] = 0.55
            valence = 0.08
            arousal = 0.52
            text = "梦里，她没有哭，而是安静地后退半步，说：这一次我要先保护自己。"
        elif operator == "guarded_projection":
            trait_evidence["vigilance"] = clamp(trait_evidence.get("vigilance", 0.0) + 0.50, -1.0, 1.0)
            trait_evidence["defensiveness"] = clamp(trait_evidence.get("defensiveness", 0.0) + 0.45, -1.0, 1.0)
            trait_evidence["trust"] = clamp(trait_evidence.get("trust", 0.0) - 0.45, -1.0, 1.0)
            belief_evidence["others_reliable"] = -0.65
            belief_evidence["vulnerability_safe"] = -0.20
            valence = -0.22
            arousal = 0.58
            text = "梦里，她提前预见了伤害，于是把心门关得很早，仿佛这样就不会再被刺痛。"
        elif operator == "repair_rehearsal":
            trait_evidence["trust"] = clamp(trait_evidence.get("trust", 0.0) + 0.28, -1.0, 1.0)
            trait_evidence["defensiveness"] = clamp(trait_evidence.get("defensiveness", 0.0) - 0.18, -1.0, 1.0)
            trait_evidence["coherence"] = clamp(trait_evidence.get("coherence", 0.0) + 0.25, -1.0, 1.0)
            belief_evidence["vulnerability_safe"] = 0.30
            belief_evidence["others_reliable"] = 0.18
            valence = 0.18
            arousal = 0.28
            text = "梦里，曾经的伤口被重新走过一遍，但这次有人停下来，接住了她没有说出口的情绪。"
        elif operator == "integration_bridge":
            trait_evidence["autonomy"] = clamp(trait_evidence.get("autonomy", 0.0) + 0.22, -1.0, 1.0)
            trait_evidence["trust"] = clamp(trait_evidence.get("trust", 0.0) + 0.14, -1.0, 1.0)
            trait_evidence["defensiveness"] = clamp(trait_evidence.get("defensiveness", 0.0) - 0.16, -1.0, 1.0)
            trait_evidence["coherence"] = clamp(trait_evidence.get("coherence", 0.0) + 0.36, -1.0, 1.0)
            belief_evidence["boundaries_allowed"] = 0.28
            belief_evidence["vulnerability_safe"] = 0.24
            valence = 0.12
            arousal = 0.36
            text = "梦里，两种自己终于站在同一条走廊上：一个想靠近，一个想防御，但她们没有再互相否认。"
        else:
            trait_evidence["openness"] = clamp(trait_evidence.get("openness", 0.0) + 0.18, -1.0, 1.0)
            trait_evidence["coherence"] = clamp(trait_evidence.get("coherence", 0.0) + 0.12, -1.0, 1.0)
            valence = 0.04
            arousal = 0.24
            text = "梦里，碎片之间长出一条新的路，她还不知道答案，但已经看见了另一种可能。"

        frame = EventFrame(
            trait_evidence=trait_evidence,
            belief_evidence=belief_evidence,
            valence=valence,
            arousal=arousal,
            tags=tuple(sorted(tags)),
            threat=max(0.0, -trait_evidence.get("trust", 0.0)),
            care=max(0.0, trait_evidence.get("trust", 0.0) * 0.5),
            control=max(0.0, trait_evidence.get("defensiveness", 0.0) * 0.25),
            abandonment=max(0.0, trait_evidence.get("attachment", 0.0) * 0.2),
            agency=max(0.0, trait_evidence.get("autonomy", 0.0) + trait_evidence.get("assertiveness", 0.0)),
            shame=max(0.0, -trait_evidence.get("coherence", 0.0)),
        )
        resonance = float(np.dot(frame.trait_vector(), state.signed_traits()))
        return text, frame, resonance

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


@dataclass(frozen=True)
class SoulMemoryConfig:
    dim: int = 384
    metric_rank: int = 64
    max_plots: int = 5000
    kde_reservoir: int = 4096
    subconscious_reservoir: int = 1024
    story_alpha: float = 1.0
    theme_alpha: float = 0.6
    gate_feature_dim: int = 8
    retrieval_kinds: Tuple[str, ...] = ("theme", "story", "plot")
    phase_refractory_steps: int = 4
    initial_archetype: str = "dependent_child"

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
            "phase_refractory_steps": self.phase_refractory_steps,
            "initial_archetype": self.initial_archetype,
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "SoulMemoryConfig":
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
            phase_refractory_steps=int(data.get("phase_refractory_steps", 4)),
            initial_archetype=str(data.get("initial_archetype", "dependent_child")),
        )


class AuroraSoulMemory:
    def __init__(
        self,
        cfg: SoulMemoryConfig = SoulMemoryConfig(),
        *,
        seed: int = 0,
        embedder: Optional[EmbeddingProvider] = None,
        extractor: Optional[MeaningExtractor] = None,
    ) -> None:
        self.cfg = cfg
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        self.embedder = embedder or HashEmbedding(dim=cfg.dim, seed=seed)
        self.extractor = extractor or HeuristicMeaningExtractor()
        self.kde = OnlineKDE(dim=cfg.dim, reservoir=cfg.kde_reservoir, seed=seed)
        self.metric = LowRankMetric(dim=cfg.dim, rank=cfg.metric_rank, seed=seed)
        self.gate = ThompsonBernoulliGate(feature_dim=cfg.gate_feature_dim, seed=seed)

        self.graph = MemoryGraph()
        self.vindex = VectorIndex(dim=cfg.dim)
        self.retriever = FieldRetriever(metric=self.metric, vindex=self.vindex, graph=self.graph)

        self.plots: Dict[str, Plot] = {}
        self.stories: Dict[str, StoryArc] = {}
        self.themes: Dict[str, Theme] = {}

        self.crp_story = CRPAssigner(alpha=cfg.story_alpha, seed=seed)
        self.story_model = StoryModel(metric=self.metric)
        self.crp_theme = CRPAssigner(alpha=cfg.theme_alpha, seed=seed + 1)
        self.theme_model = ThemeModel(metric=self.metric)

        self.subconscious = SubconsciousField(reservoir=cfg.subconscious_reservoir, seed=seed)
        self.identity = self._make_initial_identity(cfg.initial_archetype)
        self.phase_profiles = self._phase_profiles()
        self.step = 0

        self._recent_encoded_plot_ids: List[str] = []
        self.wake_evidence_pos: Dict[str, float] = {name: 1.0 for name in TRAIT_ORDER}
        self.wake_evidence_neg: Dict[str, float] = {name: 1.0 for name in TRAIT_ORDER}

    def _make_initial_identity(self, archetype: str) -> IdentityState:
        if archetype == "dependent_child":
            state = IdentityState(
                traits={
                    "attachment": 0.84,
                    "autonomy": 0.22,
                    "trust": 0.82,
                    "vigilance": 0.18,
                    "openness": 0.84,
                    "defensiveness": 0.16,
                    "assertiveness": 0.18,
                    "coherence": 0.74,
                },
                beliefs={
                    "closeness_safe": 0.82,
                    "others_reliable": 0.80,
                    "boundaries_allowed": 0.20,
                    "independence_safe": 0.22,
                    "vulnerability_safe": 0.82,
                },
                phase="dependent_child",
            )
            state.narrative_log.append("初始相位：dependent_child")
            return state
        if archetype == "guarded_teen":
            state = IdentityState(
                traits={
                    "attachment": 0.46,
                    "autonomy": 0.72,
                    "trust": 0.24,
                    "vigilance": 0.80,
                    "openness": 0.44,
                    "defensiveness": 0.78,
                    "assertiveness": 0.74,
                    "coherence": 0.58,
                },
                beliefs={
                    "closeness_safe": 0.36,
                    "others_reliable": 0.18,
                    "boundaries_allowed": 0.82,
                    "independence_safe": 0.78,
                    "vulnerability_safe": 0.18,
                },
                phase="guarded_teen",
            )
            state.narrative_log.append("初始相位：guarded_teen")
            return state
        raise ValueError(f"unknown archetype: {archetype}")

    def _phase_profiles(self) -> Dict[str, PhaseProfile]:
        return {
            "dependent_child": PhaseProfile(
                name="dependent_child",
                prototype={
                    "attachment": 0.86,
                    "autonomy": 0.22,
                    "trust": 0.84,
                    "vigilance": 0.16,
                    "openness": 0.84,
                    "defensiveness": 0.16,
                    "assertiveness": 0.18,
                    "coherence": 0.76,
                },
                barrier=0.95,
                hysteresis=0.28,
                description="依赖、柔软、相信靠近是安全的。",
            ),
            "guarded_teen": PhaseProfile(
                name="guarded_teen",
                prototype={
                    "attachment": 0.42,
                    "autonomy": 0.76,
                    "trust": 0.22,
                    "vigilance": 0.84,
                    "openness": 0.42,
                    "defensiveness": 0.82,
                    "assertiveness": 0.76,
                    "coherence": 0.56,
                },
                barrier=1.10,
                hysteresis=0.32,
                description="受伤后的自我保护，独立但带刺。",
            ),
            "exploratory_youth": PhaseProfile(
                name="exploratory_youth",
                prototype={
                    "attachment": 0.56,
                    "autonomy": 0.78,
                    "trust": 0.52,
                    "vigilance": 0.44,
                    "openness": 0.78,
                    "defensiveness": 0.34,
                    "assertiveness": 0.68,
                    "coherence": 0.68,
                },
                barrier=1.05,
                hysteresis=0.30,
                description="开始把独立和好奇同时保留下来。",
            ),
            "integrated_self": PhaseProfile(
                name="integrated_self",
                prototype={
                    "attachment": 0.58,
                    "autonomy": 0.78,
                    "trust": 0.76,
                    "vigilance": 0.32,
                    "openness": 0.78,
                    "defensiveness": 0.24,
                    "assertiveness": 0.74,
                    "coherence": 0.86,
                },
                barrier=1.20,
                hysteresis=0.36,
                description="既能亲近，也能设界；既有柔软，也有主心骨。",
            ),
        }

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
                best_sim = sim
                best_story = story
        if best_story is None:
            return 1.0
        return 1.0 - best_sim

    def _belief_support_to_traits(self, belief_name: str, value: float) -> Dict[str, float]:
        signed = signed01(value)
        mapping = {
            "closeness_safe": {"attachment": 0.45 * signed, "trust": 0.20 * signed, "vigilance": -0.18 * signed},
            "others_reliable": {"trust": 0.50 * signed, "vigilance": -0.30 * signed, "defensiveness": -0.18 * signed},
            "boundaries_allowed": {"autonomy": 0.28 * signed, "assertiveness": 0.42 * signed, "defensiveness": 0.08 * signed},
            "independence_safe": {"autonomy": 0.52 * signed, "attachment": -0.16 * signed, "coherence": 0.15 * signed},
            "vulnerability_safe": {"openness": 0.34 * signed, "trust": 0.18 * signed, "defensiveness": -0.24 * signed},
        }
        return mapping.get(belief_name, {})

    def _homeostatic_targets(self) -> Dict[str, float]:
        profile = self.phase_profiles[self.identity.phase]
        targets = {key: float(value) for key, value in profile.prototype.items()}
        for belief_name, belief_value in self.identity.beliefs.items():
            for trait, delta in self._belief_support_to_traits(belief_name, belief_value).items():
                targets[trait] = clamp(targets.get(trait, 0.5) + 0.35 * delta)
        for trait, intuition in self.identity.intuition.items():
            targets[trait] = clamp(targets.get(trait, 0.5) + 0.15 * intuition)
        return targets

    def _dissonance(self, frame: EventFrame) -> DissonanceReport:
        trait_conflicts: Dict[str, float] = {}
        trait_alignment: Dict[str, float] = {}
        for trait, evidence in frame.trait_evidence.items():
            if abs(evidence) <= 1e-9:
                continue
            align = float(evidence * signed01(self.identity.traits[trait]))
            trait_alignment[trait] = align
            trait_conflicts[trait] = max(0.0, -align)

        belief_conflicts: Dict[str, float] = {}
        belief_alignment: Dict[str, float] = {}
        for belief, evidence in frame.belief_evidence.items():
            if abs(evidence) <= 1e-9:
                continue
            align = float(evidence * signed01(self.identity.beliefs[belief]))
            belief_alignment[belief] = align
            belief_conflicts[belief] = max(0.0, -align)

        affective_load = max(0.0, -frame.valence) * (0.40 + 0.60 * frame.arousal)
        total = (
            0.45 * mean_abs(trait_conflicts.values())
            + 0.35 * mean_abs(belief_conflicts.values())
            + 0.20 * affective_load
        )
        return DissonanceReport(
            trait_conflicts=trait_conflicts,
            belief_conflicts=belief_conflicts,
            trait_alignment=trait_alignment,
            belief_alignment=belief_alignment,
            affective_load=affective_load,
            total=total,
        )

    def _voi_features(self, plot: Plot) -> np.ndarray:
        return np.asarray(
            [
                plot.surprise,
                plot.pred_error,
                1.0 - plot.redundancy,
                plot.goal_relevance,
                plot.contradiction,
                plot.frame.arousal,
                1.0 if plot.source == "wake" else 0.5,
                1.0,
            ],
            dtype=np.float32,
        )

    def _assimilation(self, plot: Plot, dissonance: DissonanceReport) -> None:
        eta = 0.04 * plot.evidence_weight * (0.35 + self.identity.plasticity())
        for trait, evidence in plot.frame.trait_evidence.items():
            if dissonance.trait_alignment.get(trait, 0.0) > 0.0:
                self.identity.traits[trait] = clamp(self.identity.traits[trait] + eta * evidence)
        for belief, evidence in plot.frame.belief_evidence.items():
            if dissonance.belief_alignment.get(belief, 0.0) > 0.0:
                self.identity.beliefs[belief] = clamp(self.identity.beliefs[belief] + eta * evidence)

    def _record_wake_evidence(self, frame: EventFrame) -> None:
        for trait, evidence in frame.trait_evidence.items():
            if evidence > 0:
                self.wake_evidence_pos[trait] += abs(evidence)
            elif evidence < 0:
                self.wake_evidence_neg[trait] += abs(evidence)

    def ingest(
        self,
        interaction_text: str,
        *,
        actors: Optional[Sequence[str]] = None,
        context_text: Optional[str] = None,
        source: Literal["wake", "dream", "repair", "phase"] = "wake",
        confidence: float = 1.0,
        evidence_weight: float = 1.0,
        ts: Optional[float] = None,
        plot_id: Optional[str] = None,
    ) -> Plot:
        self.step += 1
        event_ts = ts or now_ts()
        frame = self.extractor.extract(interaction_text)
        emb = self.embedder.embed(interaction_text)
        plot = Plot(
            id=plot_id or str(uuid.uuid4()),
            ts=event_ts,
            text=interaction_text,
            actors=tuple(actors) if actors else ("user", "agent"),
            embedding=emb,
            frame=frame,
            source=source,
            confidence=confidence,
            evidence_weight=evidence_weight,
        )
        self.kde.add(emb)
        context_emb = self.embedder.embed(context_text) if context_text else None
        plot.surprise = float(self.kde.surprise(emb))
        plot.pred_error = float(self._pred_error(emb))
        plot.redundancy = float(self._redundancy(emb))
        plot.goal_relevance = float(self._goal_relevance(emb, context_emb))

        dissonance = self._dissonance(frame)
        plot.contradiction = float(dissonance.total)
        plot.tension = float(
            0.35 * plot.surprise
            + 0.30 * plot.pred_error
            + 0.20 * plot.contradiction
            + 0.15 * max(0.0, -frame.valence) * (0.4 + frame.arousal)
        )

        self._assimilation(plot, dissonance)
        if source == "wake":
            self._record_wake_evidence(frame)

        energy_scale = {"wake": 1.0, "dream": 0.35, "repair": 0.20, "phase": 0.15}[source]
        self.identity.active_energy += energy_scale * plot.tension
        self.identity.contradiction_ema = 0.85 * self.identity.contradiction_ema + 0.15 * energy_scale * plot.contradiction

        features = self._voi_features(plot)
        encode = source == "wake" and not self.plots
        if encode or self.gate.decide(features):
            self._store_plot(plot)
            self._recent_encoded_plot_ids.append(plot.id)
            if len(self._recent_encoded_plot_ids) > 250:
                self._recent_encoded_plot_ids = self._recent_encoded_plot_ids[-250:]

        self.subconscious.deposit(plot, unresolved=plot.contradiction)
        if source == "wake":
            self._maybe_reconstruct(plot, dissonance)
            self._maybe_phase_transition(plot)

        self._pressure_manage()
        return plot

    def _store_plot(self, plot: Plot) -> None:
        logps: Dict[str, float] = {}
        for story_id, story in self.stories.items():
            prior = math.log(len(story.plot_ids) + 1e-6)
            logps[story_id] = prior + self.story_model.loglik(plot, story)

        chosen, _ = self.crp_story.sample(logps)
        if chosen is None:
            story = StoryArc(id=str(uuid.uuid4()), created_ts=plot.ts, updated_ts=plot.ts)
            self.stories[story.id] = story
            self.graph.add_node(story.id, "story", story)
            self.vindex.add(story.id, plot.embedding, kind="story")
            chosen = story.id

        story = self.stories[chosen]
        if story.centroid is not None:
            story._update_stats("dist", self.metric.d2(plot.embedding, story.centroid))
            story._update_stats("gap", max(0.0, plot.ts - story.updated_ts))
        story.plot_ids.append(plot.id)
        story.updated_ts = plot.ts
        story.unresolved_energy += plot.contradiction
        story.source_counts[plot.source] = story.source_counts.get(plot.source, 0) + 1
        for actor in plot.actors:
            story.actor_counts[actor] = story.actor_counts.get(actor, 0) + 1
        story.tension_curve.append(plot.tension)
        if story.centroid is None:
            story.centroid = plot.embedding.copy()
        else:
            n = len(story.plot_ids)
            story.centroid = l2_normalize(story.centroid * ((n - 1) / n) + plot.embedding * (1.0 / n))

        plot.story_id = story.id
        self.plots[plot.id] = plot
        self.graph.add_node(plot.id, "plot", plot)
        self.vindex.add(plot.id, plot.embedding, kind="plot")
        self.graph.ensure_edge(plot.id, story.id, "belongs_to")
        self.graph.ensure_edge(story.id, plot.id, "contains")
        if len(story.plot_ids) > 1:
            prev = story.plot_ids[-2]
            self.graph.ensure_edge(prev, plot.id, "temporal")
        for plot_id, _sim in self.vindex.search(plot.embedding, k=8, kind="plot"):
            if plot_id == plot.id:
                continue
            self.graph.ensure_edge(plot.id, plot_id, "semantic")
            self.graph.ensure_edge(plot_id, plot.id, "semantic")

    def _dynamic_repair_threshold(self) -> float:
        return float(
            1.10
            + 0.75 * self.identity.traits["coherence"]
            + 0.35 * self.identity.rigidity()
            - 0.30 * self.identity.plasticity()
        )

    def _candidate_repairs(self, plot: Plot, dissonance: DissonanceReport) -> List[CandidateRepair]:
        state = self.identity
        frame = plot.frame
        targets = self._homeostatic_targets()
        repeated_conflict = clamp(state.contradiction_ema + 0.35 * state.repressed_energy, 0.0, 1.5)

        def clone_state() -> Tuple[Dict[str, float], Dict[str, float]]:
            return copy.deepcopy(state.traits), copy.deepcopy(state.beliefs)

        def move_toward(traits: Dict[str, float], rate: float) -> None:
            for trait, evidence in frame.trait_evidence.items():
                traits[trait] = clamp(traits[trait] + rate * evidence)

        def move_beliefs(beliefs: Dict[str, float], rate: float) -> None:
            for belief, evidence in frame.belief_evidence.items():
                beliefs[belief] = clamp(beliefs[belief] + rate * evidence)

        def score_candidate(
            mode: str,
            traits: Dict[str, float],
            beliefs: Dict[str, float],
            active_after: float,
            repressed_after: float,
            explanation: str,
        ) -> CandidateRepair:
            drift = mean_abs(traits[t] - state.traits[t] for t in TRAIT_ORDER) + mean_abs(
                beliefs[b] - state.beliefs[b] for b in BELIEF_ORDER
            )
            coherence_gain = traits["coherence"] - state.traits["coherence"]
            target_distance = mean_abs(traits[t] - targets[t] for t in TRAIT_ORDER)
            fragility = (
                0.55 * traits["defensiveness"]
                + 0.30 * traits["vigilance"]
                + 0.15 * (1.0 - traits["coherence"])
            )
            dream_alignment = 0.0
            for trait in TRAIT_ORDER:
                dream_alignment += self.identity.intuition.get(trait, 0.0) * signed01(traits[trait])
            dream_alignment /= len(TRAIT_ORDER)

            utility = -(
                active_after
                + 0.60 * repressed_after
                + 0.35 * target_distance
                + 0.28 * fragility
                + 0.25 * drift * (1.0 + state.rigidity())
            ) + 0.32 * coherence_gain + 0.18 * dream_alignment

            if mode == "boundary":
                utility += 0.18 * (frame.threat + frame.control + frame.abandonment)
            if mode == "defend":
                utility += 0.10 * state.rigidity() + 0.08 * frame.threat
            if mode == "revise":
                utility += 0.12 * repeated_conflict + 0.10 * plot.evidence_weight
            if mode == "integrate":
                utility += 0.14 * state.plasticity() + 0.12 * abs(dream_alignment)

            return CandidateRepair(
                mode=mode,
                new_traits=traits,
                new_beliefs=beliefs,
                active_energy_after=max(0.0, active_after),
                repressed_energy_after=max(0.0, repressed_after),
                identity_drift=drift,
                coherence_gain=coherence_gain,
                utility=utility,
                explanation=explanation,
            )

        candidates: List[CandidateRepair] = []

        traits, beliefs = clone_state()
        traits["defensiveness"] = clamp(traits["defensiveness"] + 0.12 * frame.threat + 0.08 * frame.control)
        traits["vigilance"] = clamp(traits["vigilance"] + 0.10 * frame.threat)
        traits["trust"] = clamp(traits["trust"] - 0.08 * frame.threat)
        candidates.append(
            score_candidate(
                "defend",
                traits,
                beliefs,
                state.active_energy - 0.35 * dissonance.total,
                state.repressed_energy + 0.55 * dissonance.total,
                "这不是我变了，是外界暂时不配让我继续那样毫无防备。",
            )
        )

        traits, beliefs = clone_state()
        move_toward(traits, rate=0.18 * plot.evidence_weight * (0.5 + state.plasticity()))
        move_beliefs(beliefs, rate=0.14 * plot.evidence_weight * (0.5 + state.plasticity()))
        traits["coherence"] = clamp(traits["coherence"] + 0.04)
        candidates.append(
            score_candidate(
                "reframe",
                traits,
                beliefs,
                state.active_energy - 0.55 * dissonance.total,
                state.repressed_energy + 0.15 * dissonance.total,
                "也许我不必立刻否定自己，只需要换一种理解这件事的方式。",
            )
        )

        traits, beliefs = clone_state()
        revise_rate = 0.28 * plot.evidence_weight * (0.35 + repeated_conflict + state.plasticity())
        move_toward(traits, rate=revise_rate)
        move_beliefs(beliefs, rate=0.35 * revise_rate)
        traits["coherence"] = clamp(traits["coherence"] + 0.06)
        candidates.append(
            score_candidate(
                "revise",
                traits,
                beliefs,
                state.active_energy - 0.78 * dissonance.total,
                state.repressed_energy + 0.06 * dissonance.total,
                "如果现实反复打碎旧叙事，我就必须承认：我正在变成新的自己。",
            )
        )

        traits, beliefs = clone_state()
        rel = frame.threat + frame.control + frame.abandonment
        traits["autonomy"] = clamp(traits["autonomy"] + 0.20 * rel)
        traits["assertiveness"] = clamp(traits["assertiveness"] + 0.24 * rel)
        traits["attachment"] = clamp(traits["attachment"] - 0.12 * rel)
        traits["defensiveness"] = clamp(traits["defensiveness"] + 0.12 * rel)
        traits["trust"] = clamp(traits["trust"] - 0.10 * frame.threat)
        traits["coherence"] = clamp(traits["coherence"] + 0.02 * rel)
        beliefs["boundaries_allowed"] = clamp(beliefs["boundaries_allowed"] + 0.34 * rel)
        beliefs["independence_safe"] = clamp(beliefs["independence_safe"] + 0.22 * rel)
        candidates.append(
            score_candidate(
                "boundary",
                traits,
                beliefs,
                state.active_energy - 0.66 * dissonance.total,
                state.repressed_energy + 0.10 * dissonance.total,
                "我不是不需要亲近，我只是不能再没有边界。",
            )
        )

        traits, beliefs = clone_state()
        support = 0.25 + 0.50 * state.plasticity() + 0.08 * min(state.dream_count, 4)
        move_toward(traits, rate=0.10 * support)
        move_beliefs(beliefs, rate=0.08 * support)
        traits["autonomy"] = clamp(traits["autonomy"] + 0.12 * (frame.threat + frame.agency))
        traits["trust"] = clamp(traits["trust"] + 0.10 * frame.care - 0.04 * frame.threat)
        traits["defensiveness"] = clamp(traits["defensiveness"] - 0.14 * frame.care)
        traits["coherence"] = clamp(traits["coherence"] + 0.12)
        traits["openness"] = clamp(traits["openness"] + 0.08)
        beliefs["boundaries_allowed"] = clamp(max(beliefs["boundaries_allowed"], 0.50) + 0.16 * frame.threat)
        beliefs["vulnerability_safe"] = clamp(beliefs["vulnerability_safe"] + 0.12 * frame.care - 0.06 * frame.threat)
        candidates.append(
            score_candidate(
                "integrate",
                traits,
                beliefs,
                state.active_energy - 0.85 * dissonance.total,
                state.repressed_energy - 0.10 * min(state.repressed_energy, dissonance.total),
                "我可以既柔软又有边界，不必在依赖和防御之间永远二选一。",
            )
        )
        return candidates

    def _apply_candidate_repair(self, candidate: CandidateRepair, plot: Plot) -> None:
        self.identity.traits = copy.deepcopy(candidate.new_traits)
        self.identity.beliefs = copy.deepcopy(candidate.new_beliefs)
        self.identity.active_energy = candidate.active_energy_after
        self.identity.repressed_energy = candidate.repressed_energy_after
        self.identity.repair_count += 1
        self.identity.narrative_log.append(candidate.explanation)
        repair_text = (
            f"[{candidate.mode}] {candidate.explanation} "
            f"(tension={plot.tension:.3f}, contradiction={plot.contradiction:.3f})"
        )
        self.ingest(
            repair_text,
            actors=("self",),
            source="repair",
            confidence=0.72,
            evidence_weight=0.35,
        )

    def _maybe_reconstruct(self, plot: Plot, dissonance: DissonanceReport) -> None:
        pressure = self.identity.narrative_pressure()
        threshold = self._dynamic_repair_threshold()
        p_repair = sigmoid((pressure + 1.20 * dissonance.total) - threshold)
        if pressure <= threshold and self.rng.random() >= p_repair:
            self.identity.repressed_energy += 0.30 * dissonance.total
            return
        candidates = self._candidate_repairs(plot, dissonance)
        logits = [candidate.utility for candidate in candidates]
        probs = np.asarray(
            self._softmax(logits, temperature=max(0.40, 1.05 - 0.45 * self.identity.plasticity())),
            dtype=np.float64,
        )
        idx = int(self.rng.choice(np.arange(len(candidates)), p=probs))
        self._apply_candidate_repair(candidates[idx], plot)

    @staticmethod
    def _softmax(logits: Sequence[float], temperature: float = 1.0) -> List[float]:
        scaled = [float(item) / max(float(temperature), 1e-6) for item in logits]
        m = max(scaled)
        exps = [math.exp(item - m) for item in scaled]
        total = sum(exps) + 1e-12
        return [item / total for item in exps]

    def _dream_update_intuition(self, frame: EventFrame, resonance: float) -> None:
        gain = 0.10 + 0.12 * sigmoid(resonance)
        for trait, evidence in frame.trait_evidence.items():
            self.identity.intuition[trait] = float(0.86 * self.identity.intuition[trait] + 0.14 * gain * evidence)

    def _promote_dream_intuitions(self) -> None:
        for trait in TRAIT_ORDER:
            bias = self.identity.intuition[trait]
            if abs(bias) < 1e-6:
                continue
            pos = self.wake_evidence_pos[trait]
            neg = self.wake_evidence_neg[trait]
            corroboration = pos / (pos + neg + 1e-9) if bias > 0 else neg / (pos + neg + 1e-9)
            strength = abs(bias) * corroboration
            p_promote = sigmoid(3.0 * (strength - 0.35))
            if self.rng.random() < p_promote:
                self.identity.traits[trait] = clamp(self.identity.traits[trait] + 0.05 * bias)
                if trait in ("autonomy", "assertiveness"):
                    self.identity.beliefs["boundaries_allowed"] = clamp(self.identity.beliefs["boundaries_allowed"] + 0.03 * max(0.0, bias))
                if trait == "trust":
                    self.identity.beliefs["others_reliable"] = clamp(self.identity.beliefs["others_reliable"] + 0.03 * bias)

    def dream(self, n: int = 2) -> List[Plot]:
        dream_plots: List[Plot] = []
        for _ in range(n):
            frags = self.subconscious.sample(n=int(self.rng.integers(1, 4)))
            if not frags:
                continue
            text, frame, resonance = self.subconscious.synthesize(frags, self.identity)
            plot = self.ingest(
                text,
                actors=("self",),
                source="dream",
                confidence=0.34,
                evidence_weight=0.20,
            )
            self._dream_update_intuition(frame, resonance)
            self.identity.dream_count += 1
            dream_plots.append(plot)
        if dream_plots:
            self._promote_dream_intuitions()
        return dream_plots

    def _phase_energy(self, state: IdentityState, profile: PhaseProfile) -> float:
        dist = 0.0
        for trait in TRAIT_ORDER:
            delta = state.traits[trait] - profile.prototype[trait]
            dist += delta * delta
        dist /= len(TRAIT_ORDER)
        pressure_term = 0.18 * state.narrative_pressure()
        if profile.name == "guarded_teen":
            pressure_term -= 0.22 * (state.repressed_energy + state.traits["vigilance"] + state.traits["defensiveness"])
        elif profile.name == "integrated_self":
            pressure_term -= 0.16 * (state.traits["coherence"] + state.plasticity() + 0.10 * min(state.dream_count, 6))
        elif profile.name == "exploratory_youth":
            pressure_term -= 0.14 * (state.traits["autonomy"] + state.traits["openness"])
        elif profile.name == "dependent_child":
            pressure_term -= 0.18 * (state.traits["attachment"] + state.traits["trust"])
        return dist + pressure_term

    def _maybe_phase_transition(self, trigger_plot: Plot) -> Optional[str]:
        state = self.identity
        if (self.step - state.last_phase_step) < self.cfg.phase_refractory_steps:
            return None
        current = self.phase_profiles[state.phase]
        energies = {name: self._phase_energy(state, profile) for name, profile in self.phase_profiles.items()}
        best_name = min(energies.keys(), key=lambda name: energies[name])
        if best_name == state.phase:
            return None
        margin = energies[state.phase] - energies[best_name]
        transition_force = margin + 0.45 * min(state.narrative_pressure(), 4.0) + 0.08 * min(state.repair_count, 8) + 0.04 * min(state.dream_count, 8)
        dynamic_barrier = current.barrier + current.hysteresis
        if transition_force <= dynamic_barrier:
            return None
        new_profile = self.phase_profiles[best_name]
        rho = sigmoid(transition_force - dynamic_barrier)
        for trait in TRAIT_ORDER:
            state.traits[trait] = clamp((1.0 - rho) * state.traits[trait] + rho * new_profile.prototype[trait])
        if best_name == "guarded_teen":
            state.beliefs["others_reliable"] = clamp(state.beliefs["others_reliable"] - 0.20 * rho)
            state.beliefs["boundaries_allowed"] = clamp(state.beliefs["boundaries_allowed"] + 0.22 * rho)
            phase_text = "在长久的拉扯后，她忽然学会了先把自己收回来。柔软没有消失，但外面长出了一层带刺的壳。"
        elif best_name == "integrated_self":
            state.beliefs["others_reliable"] = clamp(state.beliefs["others_reliable"] + 0.16 * rho)
            state.beliefs["vulnerability_safe"] = clamp(state.beliefs["vulnerability_safe"] + 0.16 * rho)
            phase_text = "某个深夜之后，她不再把脆弱与边界看成敌人。她终于能同时握住两者。"
        elif best_name == "exploratory_youth":
            phase_text = "她开始愿意试着走出去，不再只用受伤前后的自己来定义未来。"
        else:
            phase_text = "她又暂时回到更依赖、更相信亲近的相位。"
        state.phase = best_name
        state.last_phase_step = self.step
        state.last_phase_change_ts = now_ts()
        state.active_energy *= 0.68
        state.repressed_energy *= 0.92
        state.narrative_log.append(f"相变: {best_name} — {new_profile.description}")
        self.ingest(
            f"[phase:{best_name}] {phase_text}",
            actors=("self",),
            source="phase",
            confidence=0.80,
            evidence_weight=0.30,
        )
        return best_name

    def _story_status_update(self) -> None:
        for story in self.stories.values():
            if story.status != "developing":
                continue
            if self.rng.random() < story.activity_probability():
                continue
            if len(story.tension_curve) >= 3:
                slope = story.tension_curve[-1] - story.tension_curve[0]
                p_resolve = sigmoid(-slope + 0.15 * story.source_purity())
            else:
                p_resolve = 0.50
            story.status = "resolved" if self.rng.random() < p_resolve else "abandoned"

    def _theme_name(self, story: StoryArc) -> Tuple[str, str, str]:
        source_mix = story.source_counts
        if source_mix.get("dream", 0) > source_mix.get("wake", 0):
            return ("梦中预演", "反事实或预演性整合", "pattern")
        if story.unresolved_energy > 1.25:
            return ("自我辩护回路", "为维持叙事一致性而形成的防御结构", "self")
        if source_mix.get("phase", 0) > 0:
            return ("人格相变痕迹", "由累积张力触发的相位跃迁", "self")
        return ("关系主题", "围绕亲近、边界与信任的持续母题", "pattern")

    def _emerge_themes(self) -> None:
        for story_id, story in list(self.stories.items()):
            if story.status != "resolved" or story.centroid is None:
                continue
            logps: Dict[str, float] = {}
            for theme_id, theme in self.themes.items():
                prior = math.log(len(theme.story_ids) + 1e-6)
                logps[theme_id] = prior + self.theme_model.loglik(story, theme)
            chosen, _ = self.crp_theme.sample(logps)
            if chosen is None:
                name, desc, theme_type = self._theme_name(story)
                theme = Theme(
                    id=str(uuid.uuid4()),
                    created_ts=now_ts(),
                    updated_ts=now_ts(),
                    prototype=story.centroid.copy(),
                    name=name,
                    description=desc,
                    theme_type=theme_type,  # type: ignore[arg-type]
                )
                self.themes[theme.id] = theme
                self.graph.add_node(theme.id, "theme", theme)
                self.vindex.add(theme.id, theme.prototype, kind="theme")
                chosen = theme.id
            theme = self.themes[chosen]
            if story_id not in theme.story_ids:
                theme.story_ids.append(story_id)
            theme.updated_ts = now_ts()
            if theme.prototype is None:
                theme.prototype = story.centroid.copy()
            else:
                n = len(theme.story_ids)
                theme.prototype = l2_normalize(theme.prototype * ((n - 1) / n) + story.centroid * (1.0 / n))
            self.graph.ensure_edge(story_id, theme.id, "thematizes")
            self.graph.ensure_edge(theme.id, story_id, "exemplified_by")

    def evolve(self, dreams: int = 2) -> List[Plot]:
        self._story_status_update()
        dream_plots = self.dream(n=dreams)
        if dream_plots:
            self._maybe_phase_transition(dream_plots[-1])
        self._emerge_themes()
        self._pressure_manage()
        return dream_plots

    def query(self, text: str, k: int = 5):
        trace = self.retriever.retrieve(
            query_text=text,
            embedder=self.embedder,
            state=self.identity,
            kinds=self.cfg.retrieval_kinds,
            k=k,
        )
        for node_id, _score, kind in trace.ranked:
            if kind == "plot":
                plot = self.graph.payload(node_id)
                plot.access_count += 1
                plot.last_access_ts = now_ts()
            elif kind == "story":
                story = self.graph.payload(node_id)
                story.reference_count += 1
        return trace

    def feedback_retrieval(self, query_text: str, chosen_id: str, success: bool) -> None:
        q = self.embedder.embed(query_text)
        seeds = [node_id for node_id, _ in self.vindex.search(q, k=10)]
        graph = self.graph.g
        if chosen_id in graph:
            for seed in seeds:
                if seed not in graph:
                    continue
                try:
                    path = list(nx.shortest_path(graph, source=seed, target=chosen_id))
                except Exception:
                    continue
                for src, dst in zip(path[:-1], path[1:]):
                    self.graph.edge_belief(src, dst).update(success)

        if chosen_id in graph:
            chosen = self.graph.payload(chosen_id)
            pos_emb = getattr(chosen, "embedding", getattr(chosen, "centroid", getattr(chosen, "prototype", None)))
            if pos_emb is not None:
                candidates = [node_id for node_id, _ in self.vindex.search(q, k=30) if node_id != chosen_id and node_id in graph]
                if candidates:
                    neg_id = random.choice(candidates)
                    negative = self.graph.payload(neg_id)
                    neg_emb = getattr(negative, "embedding", getattr(negative, "centroid", getattr(negative, "prototype", None)))
                    if neg_emb is not None:
                        self.metric.update_triplet(anchor=q, positive=pos_emb, negative=neg_emb)

        reward = 1.0 if success else -1.0
        for plot_id in self._recent_encoded_plot_ids[-20:]:
            plot = self.plots.get(plot_id)
            if plot is None:
                continue
            self.gate.update(self._voi_features(plot), reward)
        if chosen_id in self.themes:
            self.themes[chosen_id].update_evidence(success)

    def _pressure_manage(self) -> None:
        if len(self.plots) <= self.cfg.max_plots:
            return
        candidates = [plot for plot in self.plots.values() if plot.status == "active" and plot.story_id is not None]
        if not candidates:
            return
        masses = np.asarray(
            [
                plot.mass() * (1.0 + 0.2 * plot.contradiction) * (1.0 if plot.source != "dream" else 0.75)
                for plot in candidates
            ],
            dtype=np.float32,
        )
        logits = (-masses).tolist()
        probs = np.asarray(self._softmax(logits), dtype=np.float64)
        excess = len(self.plots) - self.cfg.max_plots
        remove_ids = set(self.rng.choice([plot.id for plot in candidates], size=excess, replace=False, p=probs))
        for plot_id in remove_ids:
            self._absorb_plot(plot_id)

    def _absorb_plot(self, plot_id: str) -> None:
        plot = self.plots.get(plot_id)
        if plot is None or plot.story_id is None:
            return
        plot.status = "absorbed"
        self.vindex.remove(plot_id)

    def snapshot_identity(self) -> IdentitySnapshot:
        return IdentitySnapshot(
            phase=self.identity.phase,
            traits=copy.deepcopy(self.identity.traits),
            beliefs=copy.deepcopy(self.identity.beliefs),
            active_energy=float(self.identity.active_energy),
            repressed_energy=float(self.identity.repressed_energy),
            contradiction_ema=float(self.identity.contradiction_ema),
            plasticity=float(self.identity.plasticity()),
            rigidity=float(self.identity.rigidity()),
            intuition=copy.deepcopy(self.identity.intuition),
            repair_count=int(self.identity.repair_count),
            dream_count=int(self.identity.dream_count),
            narrative_tail=list(self.identity.narrative_log[-8:]),
        )

    def narrative_summary(self) -> NarrativeSummary:
        return summarize_identity(self.identity, self.phase_profiles[self.identity.phase])

    def intuition_keywords(self, limit: int = 2) -> List[str]:
        tags = []
        for trait, value in sorted(self.identity.intuition.items(), key=lambda item: abs(item[1]), reverse=True):
            if abs(value) < 0.08:
                continue
            if trait == "trust":
                tags.append("隐约信任" if value > 0 else "克制防备")
            elif trait == "autonomy":
                tags.append("想守住边界" if value > 0 else "想被带着走")
            elif trait == "defensiveness":
                tags.append("小心翼翼" if value > 0 else "愿意放松")
            elif trait == "openness":
                tags.append("轻微好奇" if value > 0 else "有些退缩")
            elif trait == "attachment":
                tags.append("渴望靠近" if value > 0 else "想先退开")
            elif trait == "coherence":
                tags.append("努力把自己说清楚" if value > 0 else "心里有点乱")
            elif trait == "vigilance":
                tags.append("暗暗警觉" if value > 0 else "暂时安下心")
            elif trait == "assertiveness":
                tags.append("想把话说得更坚定" if value > 0 else "有点犹豫")
            if len(tags) >= limit:
                break
        return tags

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SOUL_MEMORY_STATE_VERSION,
            "seed": self._seed,
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
            "identity": self.identity.to_state_dict(),
            "step": int(self.step),
            "recent_encoded_plot_ids": list(self._recent_encoded_plot_ids),
            "wake_evidence_pos": {k: float(v) for k, v in self.wake_evidence_pos.items()},
            "wake_evidence_neg": {k: float(v) for k, v in self.wake_evidence_neg.items()},
        }

    @classmethod
    def from_state_dict(
        cls,
        data: Dict[str, Any],
        *,
        embedder: Optional[EmbeddingProvider] = None,
        extractor: Optional[MeaningExtractor] = None,
    ) -> "AuroraSoulMemory":
        if data.get("schema_version") != SOUL_MEMORY_STATE_VERSION:
            raise ValueError("Detected legacy memory snapshot. Aurora Soul-Memory requires a fresh data directory.")
        cfg = SoulMemoryConfig.from_state_dict(data["cfg"])
        obj = cls(cfg=cfg, seed=int(data.get("seed", 0)), embedder=embedder, extractor=extractor)
        obj.kde = OnlineKDE.from_state_dict(data["kde"])
        obj.metric = LowRankMetric.from_state_dict(data["metric"])
        obj.gate = ThompsonBernoulliGate.from_state_dict(data["gate"])
        obj.graph = MemoryGraph()
        obj.vindex = VectorIndex.from_state_dict(data["vindex"])
        obj.plots = {plot_id: Plot.from_state_dict(item) for plot_id, item in data.get("plots", {}).items()}
        obj.stories = {story_id: StoryArc.from_state_dict(item) for story_id, item in data.get("stories", {}).items()}
        obj.themes = {theme_id: Theme.from_state_dict(item) for theme_id, item in data.get("themes", {}).items()}
        for story in obj.stories.values():
            obj.graph.add_node(story.id, "story", story)
        for theme in obj.themes.values():
            obj.graph.add_node(theme.id, "theme", theme)
        for plot in obj.plots.values():
            obj.graph.add_node(plot.id, "plot", plot)
        obj.graph.restore_edges(data.get("graph", {}))
        obj.retriever = FieldRetriever(metric=obj.metric, vindex=obj.vindex, graph=obj.graph)
        obj.crp_story = CRPAssigner.from_state_dict(data["crp_story"])
        obj.story_model = StoryModel(metric=obj.metric)
        obj.crp_theme = CRPAssigner.from_state_dict(data["crp_theme"])
        obj.theme_model = ThemeModel(metric=obj.metric)
        obj.subconscious = SubconsciousField.from_state_dict(data["subconscious"])
        obj.identity = IdentityState.from_state_dict(data["identity"])
        obj.step = int(data.get("step", 0))
        obj._recent_encoded_plot_ids = [str(item) for item in data.get("recent_encoded_plot_ids", [])]
        obj.wake_evidence_pos = {k: float(v) for k, v in data.get("wake_evidence_pos", {}).items()}
        obj.wake_evidence_neg = {k: float(v) for k, v in data.get("wake_evidence_neg", {}).items()}
        return obj
