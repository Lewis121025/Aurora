"""
AURORA Memory Core
==================

Main entry point: AuroraMemory class.

Design: zero hard-coded thresholds. All decisions via Bayesian/stochastic policies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import random
import uuid

import numpy as np
import networkx as nx

from aurora.utils.math_utils import l2_normalize, cosine_sim, sigmoid, softmax
from aurora.utils.id_utils import det_id
from aurora.utils.time_utils import now_ts

from aurora.algorithms.models.plot import Plot
from aurora.algorithms.models.story import StoryArc
from aurora.algorithms.models.theme import Theme
from aurora.algorithms.models.config import MemoryConfig
from aurora.algorithms.models.trace import RetrievalTrace, EvolutionSnapshot, EvolutionPatch

from aurora.algorithms.components.density import OnlineKDE
from aurora.algorithms.components.metric import LowRankMetric
from aurora.algorithms.components.bandit import ThompsonBernoulliGate
from aurora.algorithms.components.assignment import CRPAssigner, StoryModel, ThemeModel
from aurora.algorithms.components.embedding import HashEmbedding

from aurora.algorithms.graph.memory_graph import MemoryGraph
from aurora.algorithms.graph.vector_index import VectorIndex

from aurora.algorithms.retrieval.field_retriever import FieldRetriever


class AuroraMemory:
    """AURORA Memory: emergent narrative memory from first principles.

    Key APIs:
        ingest(interaction_text, actors, context_text) -> Plot (may or may not be stored)
        query(text, k) -> RetrievalTrace
        feedback_retrieval(query_text, chosen_id, success) -> update beliefs
        evolve() -> consolidate plots->stories->themes, manage pressure, update statuses
    """

    def __init__(self, cfg: MemoryConfig = MemoryConfig(), seed: int = 0):
        self.cfg = cfg
        self._seed = seed
        self.rng = np.random.default_rng(seed)

        # learnable primitives
        self.embedder = HashEmbedding(dim=cfg.dim, seed=seed)
        self.kde = OnlineKDE(dim=cfg.dim, reservoir=cfg.kde_reservoir, seed=seed)
        self.metric = LowRankMetric(dim=cfg.dim, rank=cfg.metric_rank, seed=seed)
        self.gate = ThompsonBernoulliGate(feature_dim=cfg.gate_feature_dim, seed=seed)

        # memory stores
        self.graph = MemoryGraph()
        self.vindex = VectorIndex(dim=cfg.dim)

        self.plots: Dict[str, Plot] = {}
        self.stories: Dict[str, StoryArc] = {}
        self.themes: Dict[str, Theme] = {}

        # nonparametric assignment
        self.crp_story = CRPAssigner(alpha=cfg.story_alpha, seed=seed)
        self.story_model = StoryModel(metric=self.metric)

        self.crp_theme = CRPAssigner(alpha=cfg.theme_alpha, seed=seed + 1)
        self.theme_model = ThemeModel(metric=self.metric)

        self.retriever = FieldRetriever(metric=self.metric, vindex=self.vindex, graph=self.graph)

        # bookkeeping for delayed credit assignment
        self._recent_encoded_plot_ids: List[str] = []  # sliding window

    # -------------------------------------------------------------------------
    # Feature computation for value-of-information (VOI) encoding
    # -------------------------------------------------------------------------

    def _redundancy(self, emb: np.ndarray) -> float:
        hits = self.vindex.search(emb, k=8, kind="plot")
        return max((s for _, s in hits), default=0.0)

    def _goal_relevance(self, emb: np.ndarray, context_emb: Optional[np.ndarray]) -> float:
        return cosine_sim(emb, context_emb) if context_emb is not None else 0.0

    def _pred_error(self, emb: np.ndarray) -> float:
        # predictive error vs best-matching story centroid under current metric
        best = None
        best_sim = -1.0
        for s in self.stories.values():
            if s.centroid is None:
                continue
            sim = self.metric.sim(emb, s.centroid)
            if sim > best_sim:
                best_sim = sim
                best = s
        if best is None:
            return 1.0
        return 1.0 - best_sim

    def _voi_features(self, plot: Plot) -> np.ndarray:
        # Features are observables, not manually weighted rules.
        return np.array([
            plot.surprise,
            plot.pred_error,
            1.0 - plot.redundancy,
            plot.goal_relevance,
            math.tanh(len(plot.text) / 512.0),
            1.0,
        ], dtype=np.float32)

    # -------------------------------------------------------------------------
    # Ingest
    # -------------------------------------------------------------------------

    def ingest(
        self,
        interaction_text: str,
        actors: Optional[Sequence[str]] = None,
        context_text: Optional[str] = None,
        event_id: Optional[str] = None,
    ) -> Plot:
        """Ingest an interaction/event.

        This method:
        1) embeds the interaction
        2) computes surprise + predictive error + redundancy + relevance
        3) decides to store stochastically via Thompson-sampled encode gate
        4) if stored, assigns it into a story (CRP) and updates graph edges
        """
        actors = tuple(actors) if actors else ("user", "agent")
        emb = self.embedder.embed(interaction_text)

        plot = Plot(
            id=det_id("plot", event_id) if event_id else str(uuid.uuid4()),
            ts=now_ts(),
            text=interaction_text,
            actors=tuple(actors),
            embedding=emb,
        )

        # Update global density regardless of storage decision (calibration)
        self.kde.add(emb)

        context_emb = self.embedder.embed(context_text) if context_text else None
        plot.surprise = float(self.kde.surprise(emb))
        plot.pred_error = float(self._pred_error(emb))
        plot.redundancy = float(self._redundancy(emb))
        plot.goal_relevance = float(self._goal_relevance(emb, context_emb))

        # Free-energy proxy: surprise plus belief update magnitude (here pred_error)
        plot.tension = plot.surprise * (1.0 + plot.pred_error)

        # Decide encode stochastically (no threshold)
        x = self._voi_features(plot)
        encode = self.gate.decide(x)

        if encode:
            self._store_plot(plot)
            self._recent_encoded_plot_ids.append(plot.id)
            # keep sliding window bounded
            if len(self._recent_encoded_plot_ids) > 200:
                self._recent_encoded_plot_ids = self._recent_encoded_plot_ids[-200:]
        # else: dropped (but still influenced KDE as calibration)

        # Pressure management: consolidate/absorb when exceeding capacity (probabilistic)
        self._pressure_manage()
        return plot

    # -------------------------------------------------------------------------
    # Store plot into story + graph weaving
    # -------------------------------------------------------------------------

    def _store_plot(self, plot: Plot) -> None:
        # 1) choose story assignment probabilities via CRP posterior
        logps: Dict[str, float] = {}
        for sid, s in self.stories.items():
            # prior proportional to story size
            prior = math.log(len(s.plot_ids) + 1e-6)
            logps[sid] = prior + self.story_model.loglik(plot, s)

        chosen, post = self.crp_story.sample(logps)
        if chosen is None:
            story = StoryArc(id=det_id("story", plot.id), created_ts=now_ts(), updated_ts=now_ts())
            self.stories[story.id] = story
            self.graph.add_node(story.id, "story", story)
            self.vindex.add(story.id, plot.embedding, kind="story")
            chosen = story.id

        story = self.stories[chosen]

        # 2) update story statistics and centroid
        if story.centroid is not None:
            d2 = self.metric.d2(plot.embedding, story.centroid)
            story._update_stats("dist", d2)
            gap = max(0.0, plot.ts - story.updated_ts)
            story._update_stats("gap", gap)

        story.plot_ids.append(plot.id)
        story.updated_ts = plot.ts
        story.actor_counts = {a: story.actor_counts.get(a, 0) + 1 for a in plot.actors}
        story.tension_curve.append(plot.tension)

        if story.centroid is None:
            story.centroid = plot.embedding.copy()
        else:
            n = len(story.plot_ids)
            story.centroid = l2_normalize(story.centroid * ((n - 1) / n) + plot.embedding * (1.0 / n))

        # 3) store plot
        plot.story_id = story.id
        self.plots[plot.id] = plot
        self.graph.add_node(plot.id, "plot", plot)
        self.vindex.add(plot.id, plot.embedding, kind="plot")

        # 4) weave edges (typed relations; strength learned by Beta posteriors)
        self.graph.ensure_edge(plot.id, story.id, "belongs_to")
        self.graph.ensure_edge(story.id, plot.id, "contains")

        # temporal to previous plot in story
        if len(story.plot_ids) > 1:
            prev = story.plot_ids[-2]
            self.graph.ensure_edge(prev, plot.id, "temporal")

        # semantic links to nearest neighbors (no threshold; just top-k)
        for pid, _sim in self.vindex.search(plot.embedding, k=8, kind="plot"):
            if pid == plot.id:
                continue
            self.graph.ensure_edge(plot.id, pid, "semantic")
            self.graph.ensure_edge(pid, plot.id, "semantic")

    # -------------------------------------------------------------------------
    # Query / retrieval
    # -------------------------------------------------------------------------

    def query(self, text: str, k: int = 5) -> RetrievalTrace:
        trace = self.retriever.retrieve(query_text=text, embed=self.embedder, kinds=self.cfg.retrieval_kinds, k=k)

        # update access/mass
        for nid, _score, kind in trace.ranked:
            if kind == "plot":
                p: Plot = self.graph.payload(nid)
                p.access_count += 1
                p.last_access_ts = now_ts()
            elif kind == "story":
                s: StoryArc = self.graph.payload(nid)
                s.reference_count += 1
        return trace

    # -------------------------------------------------------------------------
    # Feedback: credit assignment and learning
    # -------------------------------------------------------------------------

    def feedback_retrieval(self, query_text: str, chosen_id: str, success: bool) -> None:
        """Delayed reward signal.

        Updates:
        - edge Beta posteriors along paths from query seeds to chosen item
        - metric using a triplet (query, chosen, negative)
        - encode gate using aggregate reward attributed to recent encoded plots
        """
        q = self.embedder.embed(query_text)

        # 1) update edges on (a few) shortest paths from seeds
        seeds = [i for i, _ in self.vindex.search(q, k=10)]
        G = self.graph.g
        if chosen_id in G:
            for s in seeds:
                if s not in G:
                    continue
                try:
                    path = nx.shortest_path(G, source=s, target=chosen_id)
                except nx.NetworkXNoPath:
                    continue
                for u, v in zip(path[:-1], path[1:]):
                    self.graph.edge_belief(u, v).update(success)

        # 2) metric triplet update
        if chosen_id in G:
            chosen = self.graph.payload(chosen_id)
            pos_emb = getattr(chosen, "embedding", getattr(chosen, "centroid", getattr(chosen, "prototype", None)))
            if pos_emb is not None:
                # pick negative among high-sim but not chosen
                cands = [i for i, _ in self.vindex.search(q, k=30) if i != chosen_id and i in G]
                if cands:
                    neg_id = random.choice(cands)
                    neg = self.graph.payload(neg_id)
                    neg_emb = getattr(neg, "embedding", getattr(neg, "centroid", getattr(neg, "prototype", None)))
                    if neg_emb is not None:
                        self.metric.update_triplet(anchor=q, positive=pos_emb, negative=neg_emb)

        # 3) encode gate update: attribute reward to recently encoded plots (soft credit)
        reward = 1.0 if success else -1.0
        # Use a light heuristic: distribute reward uniformly to last N encoded plots
        recent = self._recent_encoded_plot_ids[-20:]
        for pid in recent:
            p = self.plots.get(pid)
            if p is None:
                continue
            x = self._voi_features(p)
            self.gate.update(x, reward)

        # 4) theme evidence update if chosen was a theme
        if chosen_id in self.themes:
            self.themes[chosen_id].update_evidence(success)

    # -------------------------------------------------------------------------
    # Evolution: consolidation & theme emergence (Plot -> Story -> Theme)
    # -------------------------------------------------------------------------

    def evolve(self) -> None:
        """Offline-ish evolution step (can be triggered periodically or on idle).

        This method:
        - updates story statuses stochastically based on activity probabilities
        - assigns stories into themes via CRP (hierarchical emergence)
        - absorbs low-mass plots into their story to save space (pressure-driven)
        """
        # 1) update story statuses (probabilistic, no idle threshold)
        for s in self.stories.values():
            if s.status != "developing":
                continue
            p_active = s.activity_probability()
            # If story is still active, keep; else sample resolution vs abandonment
            if self.rng.random() < p_active:
                continue
            # resolve vs abandon depends on tension curve (if it has a decay, more likely resolved)
            if len(s.tension_curve) >= 3:
                # crude slope: last - first
                slope = s.tension_curve[-1] - s.tension_curve[0]
                p_resolve = sigmoid(-slope)  # if tension decreased, p_resolve high
            else:
                p_resolve = 0.5
            s.status = "resolved" if (self.rng.random() < p_resolve) else "abandoned"

        # 2) theme emergence: assign each resolved story into a theme
        for sid, s in list(self.stories.items()):
            if s.status != "resolved":
                continue
            if s.centroid is None:
                continue
            logps: Dict[str, float] = {}
            for tid, t in self.themes.items():
                prior = math.log(len(t.story_ids) + 1e-6)
                logps[tid] = prior + self.theme_model.loglik(s, t)
            chosen, post = self.crp_theme.sample(logps)
            if chosen is None:
                theme = Theme(id=det_id("theme", sid), created_ts=now_ts(), updated_ts=now_ts())
                theme.prototype = s.centroid.copy()
                self.themes[theme.id] = theme
                self.graph.add_node(theme.id, "theme", theme)
                self.vindex.add(theme.id, theme.prototype, kind="theme")
                chosen = theme.id
            theme = self.themes[chosen]
            theme.story_ids.append(sid)
            theme.updated_ts = now_ts()
            # prototype update (online mean)
            if theme.prototype is None:
                theme.prototype = s.centroid.copy()
            else:
                n = len(theme.story_ids)
                theme.prototype = l2_normalize(theme.prototype * ((n - 1) / n) + s.centroid * (1.0 / n))

            # weave edges
            self.graph.ensure_edge(sid, theme.id, "thematizes")
            self.graph.ensure_edge(theme.id, sid, "exemplified_by")

        # 3) absorb low-mass plots within stories to satisfy capacity (pressure-driven)
        self._pressure_manage()

    # -------------------------------------------------------------------------
    # Async Evolution: Copy-on-Write for non-blocking processing
    # -------------------------------------------------------------------------

    def create_evolution_snapshot(self) -> "EvolutionSnapshot":
        """Create a read-only snapshot for evolution processing.
        
        This captures the current state without holding locks, allowing
        evolution to proceed in a background thread while new ingests continue.
        """
        return EvolutionSnapshot(
            story_ids=list(self.stories.keys()),
            story_statuses={sid: s.status for sid, s in self.stories.items()},
            story_centroids={sid: s.centroid.copy() if s.centroid is not None else None 
                           for sid, s in self.stories.items()},
            story_tension_curves={sid: list(s.tension_curve) for sid, s in self.stories.items()},
            story_updated_ts={sid: s.updated_ts for sid, s in self.stories.items()},
            story_gap_means={sid: s.gap_mean_safe() for sid, s in self.stories.items()},
            theme_ids=list(self.themes.keys()),
            theme_story_counts={tid: len(t.story_ids) for tid, t in self.themes.items()},
            theme_prototypes={tid: t.prototype.copy() if t.prototype is not None else None
                            for tid, t in self.themes.items()},
            crp_theme_alpha=self.crp_theme.alpha,
            rng_state=self.rng.bit_generator.state,
        )

    def compute_evolution_patch(self, snapshot: "EvolutionSnapshot") -> "EvolutionPatch":
        """Compute evolution changes from snapshot (pure function, no side effects).
        
        This can run in a background thread without locks.
        """
        # Use a local RNG to avoid state mutation
        rng = np.random.default_rng()
        rng.bit_generator.state = snapshot.rng_state
        
        # 1) Determine story status changes
        status_changes: Dict[str, str] = {}
        for sid in snapshot.story_ids:
            if snapshot.story_statuses[sid] != "developing":
                continue
            
            # Activity probability
            ts = now_ts()
            updated = snapshot.story_updated_ts[sid]
            tau = snapshot.story_gap_means[sid]
            idle = max(0.0, ts - updated)
            p_active = math.exp(-idle / max(tau, 1e-6))
            
            if rng.random() < p_active:
                continue
            
            # Resolve vs abandon
            curve = snapshot.story_tension_curves[sid]
            if len(curve) >= 3:
                slope = curve[-1] - curve[0]
                p_resolve = sigmoid(-slope)
            else:
                p_resolve = 0.5
            
            new_status = "resolved" if rng.random() < p_resolve else "abandoned"
            status_changes[sid] = new_status
        
        # 2) Compute theme assignments for newly resolved stories
        theme_assignments: List[Tuple[str, Optional[str]]] = []  # (story_id, theme_id or None for new)
        new_themes: List[Tuple[str, np.ndarray]] = []  # (theme_id, prototype)
        
        # Build updated theme counts (accounting for new assignments)
        current_theme_counts = dict(snapshot.theme_story_counts)
        
        for sid, new_status in status_changes.items():
            if new_status != "resolved":
                continue
            
            centroid = snapshot.story_centroids[sid]
            if centroid is None:
                continue
            
            # Compute log probabilities for existing themes
            logps: Dict[str, float] = {}
            for tid in snapshot.theme_ids:
                prior = math.log(current_theme_counts.get(tid, 0) + 1e-6)
                prototype = snapshot.theme_prototypes.get(tid)
                if prototype is not None:
                    # Simple distance-based likelihood
                    d2 = float(np.dot(centroid - prototype, centroid - prototype))
                    logps[tid] = prior - 0.5 * d2
                else:
                    logps[tid] = prior
            
            # Add new theme option (CRP)
            logps["__new__"] = math.log(snapshot.crp_theme_alpha)
            
            # Sample
            keys = list(logps.keys())
            probs = softmax([logps[k] for k in keys])
            choice = rng.choice(keys, p=np.array(probs, dtype=np.float64))
            
            if choice == "__new__":
                # Create new theme
                new_theme_id = det_id("theme", sid)
                new_themes.append((new_theme_id, centroid.copy()))
                theme_assignments.append((sid, new_theme_id))
                current_theme_counts[new_theme_id] = 1
            else:
                theme_assignments.append((sid, choice))
                current_theme_counts[choice] = current_theme_counts.get(choice, 0) + 1
        
        return EvolutionPatch(
            status_changes=status_changes,
            theme_assignments=theme_assignments,
            new_themes=new_themes,
        )

    def apply_evolution_patch(self, patch: "EvolutionPatch") -> None:
        """Apply computed evolution changes atomically.
        
        Should be called with appropriate locking if needed.
        """
        # 1) Apply story status changes
        for sid, new_status in patch.status_changes.items():
            if sid in self.stories:
                self.stories[sid].status = new_status
        
        # 2) Create new themes
        for theme_id, prototype in patch.new_themes:
            theme = Theme(id=theme_id, created_ts=now_ts(), updated_ts=now_ts())
            theme.prototype = prototype
            self.themes[theme_id] = theme
            self.graph.add_node(theme_id, "theme", theme)
            self.vindex.add(theme_id, prototype, kind="theme")
        
        # 3) Apply theme assignments and weave edges
        for sid, tid in patch.theme_assignments:
            if sid not in self.stories or tid not in self.themes:
                continue
            
            story = self.stories[sid]
            theme = self.themes[tid]
            
            theme.story_ids.append(sid)
            theme.updated_ts = now_ts()
            
            # Update prototype (online mean)
            if story.centroid is not None:
                if theme.prototype is None:
                    theme.prototype = story.centroid.copy()
                else:
                    n = len(theme.story_ids)
                    theme.prototype = l2_normalize(
                        theme.prototype * ((n - 1) / n) + story.centroid * (1.0 / n)
                    )
            
            # Weave edges
            self.graph.ensure_edge(sid, tid, "thematizes")
            self.graph.ensure_edge(tid, sid, "exemplified_by")
        
        # 4) Pressure management
        self._pressure_manage()

    # -------------------------------------------------------------------------
    # Pressure management (resource constraints -> compression)
    # -------------------------------------------------------------------------

    def _pressure_manage(self) -> None:
        """Keep plot count under max_plots by probabilistic absorption/archival.

        No rule like 'if age>30 days' or 'if importance<0.2'. Instead:
        - compute a soft selection distribution over plots with low mass
        - sample plots to absorb/archive until within capacity
        """
        max_plots = self.cfg.max_plots
        if len(self.plots) <= max_plots:
            return

        # candidate plots that are active and already assigned to a story
        cands = [p for p in self.plots.values() if p.status == "active" and p.story_id is not None]
        if not cands:
            return

        # Build logits favoring low-mass plots
        masses = np.array([p.mass() for p in cands], dtype=np.float32)
        # lower mass -> higher probability to absorb
        logits = (-masses).tolist()
        probs = np.array(softmax(logits), dtype=np.float64)

        # number to remove
        excess = len(self.plots) - max_plots
        remove_ids = set(self.rng.choice([p.id for p in cands], size=excess, replace=False, p=probs))
        for pid in remove_ids:
            p = self.plots.get(pid)
            if p is None:
                continue
            self._absorb_plot(pid)

    def _absorb_plot(self, plot_id: str) -> None:
        """Absorb plot into its story (lose high-res detail, keep narrative)."""
        p = self.plots.get(plot_id)
        if p is None or p.story_id is None:
            return
        p.status = "absorbed"

        # Remove from vector index to reduce retrieval noise (keeps story centroid)
        self.vindex.remove(plot_id)
        # Keep node in graph for traceability, but you may remove it in production
        # self.graph.g.remove_node(plot_id)

    # -------------------------------------------------------------------------
    # Convenience: inspect
    # -------------------------------------------------------------------------

    def get_story(self, story_id: str) -> StoryArc:
        return self.stories[story_id]

    def get_plot(self, plot_id: str) -> Plot:
        return self.plots[plot_id]

    def get_theme(self, theme_id: str) -> Theme:
        return self.themes[theme_id]

    # -------------------------------------------------------------------------
    # State serialization (JSON-compatible, replaces pickle)
    # -------------------------------------------------------------------------

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize entire AuroraMemory state to JSON-compatible dict.
        
        This replaces pickle-based serialization with structured JSON,
        enabling:
        - Human-readable state inspection
        - Cross-version compatibility
        - Partial state recovery
        - State diffing and debugging
        """
        return {
            "version": 2,  # State format version for forward compatibility
            "cfg": self.cfg.to_state_dict(),
            "seed": self._seed,
            
            # Learnable components
            "kde": self.kde.to_state_dict(),
            "metric": self.metric.to_state_dict(),
            "gate": self.gate.to_state_dict(),
            
            # Nonparametric assignment
            "crp_story": self.crp_story.to_state_dict(),
            "crp_theme": self.crp_theme.to_state_dict(),
            
            # Memory data
            "plots": {pid: p.to_state_dict() for pid, p in self.plots.items()},
            "stories": {sid: s.to_state_dict() for sid, s in self.stories.items()},
            "themes": {tid: t.to_state_dict() for tid, t in self.themes.items()},
            
            # Graph structure (payloads reference plots/stories/themes)
            "graph": self.graph.to_state_dict(),
            
            # Vector index (deprecated in production, use VectorStore)
            "vindex": self.vindex.to_state_dict(),
            
            # Bookkeeping
            "recent_encoded_plot_ids": self._recent_encoded_plot_ids,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "AuroraMemory":
        """Reconstruct AuroraMemory from state dict.
        
        Handles version migration if needed.
        """
        version = d.get("version", 1)
        
        # Reconstruct config
        cfg = MemoryConfig.from_state_dict(d["cfg"])
        seed = d.get("seed", 0)
        
        # Create new instance with config
        obj = cls(cfg=cfg, seed=seed)
        
        # Restore learnable components
        obj.kde = OnlineKDE.from_state_dict(d["kde"])
        obj.metric = LowRankMetric.from_state_dict(d["metric"])
        obj.gate = ThompsonBernoulliGate.from_state_dict(d["gate"])
        
        # Restore CRP assigners
        obj.crp_story = CRPAssigner.from_state_dict(d["crp_story"])
        obj.crp_theme = CRPAssigner.from_state_dict(d["crp_theme"])
        
        # Restore memory data
        obj.plots = {pid: Plot.from_state_dict(pd) for pid, pd in d.get("plots", {}).items()}
        obj.stories = {sid: StoryArc.from_state_dict(sd) for sid, sd in d.get("stories", {}).items()}
        obj.themes = {tid: Theme.from_state_dict(td) for tid, td in d.get("themes", {}).items()}
        
        # Build payload lookup for graph reconstruction
        payloads: Dict[str, Any] = {}
        payloads.update(obj.plots)
        payloads.update(obj.stories)
        payloads.update(obj.themes)
        
        # Restore graph
        obj.graph = MemoryGraph.from_state_dict(d["graph"], payloads=payloads)
        
        # Restore vector index
        obj.vindex = VectorIndex.from_state_dict(d["vindex"])
        
        # Rebuild models with restored metric
        obj.story_model = StoryModel(metric=obj.metric)
        obj.theme_model = ThemeModel(metric=obj.metric)
        obj.retriever = FieldRetriever(metric=obj.metric, vindex=obj.vindex, graph=obj.graph)
        
        # Restore bookkeeping
        obj._recent_encoded_plot_ids = d.get("recent_encoded_plot_ids", [])
        
        return obj


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    mem = AuroraMemory(cfg=MemoryConfig(dim=96, metric_rank=32, max_plots=2000), seed=42)

    # Ingest a few interactions
    mem.ingest("用户：我想做一个记忆系统。助理：好的，我们从第一性原理开始。", context_text="memory algorithm")
    mem.ingest("用户：不要硬编码阈值。助理：可以用贝叶斯决策和随机策略。", context_text="memory algorithm")
    mem.ingest("用户：检索要能讲故事。助理：可以用故事弧 + 主题涌现。", context_text="narrative memory")
    mem.ingest("用户：给我一个可运行的实现。助理：我会给你一份python参考实现。", context_text="implementation")

    trace = mem.query("如何避免硬编码阈值并实现叙事检索？", k=5)
    print("Top results:", trace.ranked)

    # Provide feedback
    if trace.ranked:
        chosen_id = trace.ranked[0][0]
        mem.feedback_retrieval("如何避免硬编码阈值并实现叙事检索？", chosen_id=chosen_id, success=True)

    # Run evolution
    mem.evolve()
    print("stories:", len(mem.stories), "themes:", len(mem.themes), "plots:", len(mem.plots))
