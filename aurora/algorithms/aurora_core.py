"""
AURORA Memory Core
==================

Main entry point: AuroraMemory class.

Design: zero hard-coded thresholds. All decisions via Bayesian/stochastic policies.

Architecture:
- Core class inherits from specialized mixins for different concerns
- relationship.py: Relationship identification and identity assessment
- pressure.py: Growth-oriented pressure management
- evolution.py: Evolution, reflection, and meaning reframe
- serialization.py: State serialization/deserialization
"""

from __future__ import annotations

import logging
import math
import uuid
from collections import Counter, deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from aurora.algorithms.components.assignment import CRPAssigner, StoryModel, ThemeModel
from aurora.algorithms.components.bandit import ThompsonBernoulliGate
from aurora.algorithms.components.density import OnlineKDE
from aurora.algorithms.components.metric import LowRankMetric
from aurora.algorithms.constants import (
    EPSILON_PRIOR,
    EVENT_SUMMARY_MAX_LENGTH,
    IDENTITY_RELEVANCE_WEIGHT,
    MAX_RECENT_PLOTS_FOR_RETRIEVAL,
    RECENT_ENCODED_PLOTS_WINDOW,
    RECENT_PLOTS_FOR_FEEDBACK,
    RELATIONSHIP_BONUS_SCORE,
    SEMANTIC_NEIGHBORS_K,
    STORY_SIMILARITY_BONUS,
    TEXT_LENGTH_NORMALIZATION,
    TRUST_BASE,
    VOI_DECISION_WEIGHT,
)
from aurora.algorithms.evolution import EvolutionMixin
from aurora.algorithms.graph.memory_graph import MemoryGraph
from aurora.algorithms.graph.vector_index import VectorIndex
from aurora.algorithms.models.config import MemoryConfig
from aurora.algorithms.models.plot import Plot
from aurora.algorithms.models.story import StoryArc
from aurora.algorithms.models.theme import Theme
from aurora.algorithms.models.trace import RetrievalTrace
from aurora.algorithms.pressure import PressureMixin
from aurora.algorithms.relationship import RelationshipMixin
from aurora.algorithms.retrieval.field_retriever import FieldRetriever
from aurora.algorithms.serialization import SerializationMixin
from aurora.embeddings.hash import HashEmbedding
from aurora.utils.id_utils import det_id
from aurora.utils.math_utils import cosine_sim, l2_normalize, sigmoid
from aurora.utils.time_utils import now_ts

try:
    from aurora.algorithms.graph.faiss_index import FAISS_AVAILABLE, FAISSVectorIndex
except ImportError:
    FAISSVectorIndex = None
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


class AuroraMemory(RelationshipMixin, PressureMixin, EvolutionMixin, SerializationMixin):
    """AURORA Memory: emergent narrative memory from first principles.

    Key APIs:
        ingest(interaction_text, actors, context_text) -> Plot (may or may not be stored)
        query(text, k) -> RetrievalTrace
        feedback_retrieval(query_text, chosen_id, success) -> update beliefs
        evolve() -> consolidate plots->stories->themes, manage pressure, update statuses
    
    Architecture:
        This class uses mixins to separate concerns:
        - RelationshipMixin: Relationship identification and identity assessment
        - PressureMixin: Growth-oriented pressure management
        - EvolutionMixin: Evolution, reflection, and meaning reframe
        - SerializationMixin: State serialization/deserialization
    """

    def __init__(self, cfg: MemoryConfig = MemoryConfig(), seed: int = 0):
        self.cfg = cfg
        self._seed = seed
        self.rng = np.random.default_rng(seed)

        # Learnable primitives
        self.embedder = HashEmbedding(dim=cfg.dim, seed=seed)
        self.kde = OnlineKDE(dim=cfg.dim, reservoir=cfg.kde_reservoir, seed=seed)
        self.metric = LowRankMetric(dim=cfg.dim, rank=cfg.metric_rank, seed=seed)
        self.gate = ThompsonBernoulliGate(feature_dim=cfg.gate_feature_dim, seed=seed)

        # Memory stores
        self.graph = MemoryGraph()
        self.vindex = self._create_vector_index(cfg)

        self.plots: Dict[str, Plot] = {}
        self.stories: Dict[str, StoryArc] = {}
        self.themes: Dict[str, Theme] = {}

        # Nonparametric assignment
        self.crp_story = CRPAssigner(alpha=cfg.story_alpha, seed=seed)
        self.story_model = StoryModel(metric=self.metric)
        self.crp_theme = CRPAssigner(alpha=cfg.theme_alpha, seed=seed + 1)
        self.theme_model = ThemeModel(metric=self.metric)

        self.retriever = FieldRetriever(metric=self.metric, vindex=self.vindex, graph=self.graph)

        # Bookkeeping for delayed credit assignment (auto-bounded deque)
        self._recent_encoded_plot_ids: Deque[str] = deque(maxlen=RECENT_ENCODED_PLOTS_WINDOW)
        
        # Relationship-centric additions
        self._relationship_story_index: Dict[str, str] = {}  # relationship_entity -> story_id
        self._identity_dimensions: Dict[str, float] = {}  # dimension_name -> strength

    # -------------------------------------------------------------------------
    # Vector index creation
    # -------------------------------------------------------------------------

    def _create_vector_index(self, cfg: MemoryConfig) -> VectorIndex:
        """Create vector index based on configuration."""
        use_faiss = cfg.vector_backend == "faiss" or (cfg.vector_backend == "auto" and FAISS_AVAILABLE)
        if use_faiss:
            if not FAISS_AVAILABLE:
                raise ImportError("FAISS required. Install: pip install faiss-cpu")
            return FAISSVectorIndex(
                dim=cfg.dim,
                M=cfg.faiss_m,
                ef_construction=cfg.faiss_ef_construction,
                ef_search=cfg.faiss_ef_search,
            )
        return VectorIndex(dim=cfg.dim)

    # -------------------------------------------------------------------------
    # Common utility methods (extracted from repeated patterns)
    # -------------------------------------------------------------------------

    def _update_centroid_online(
        self, current: Optional[np.ndarray], new_emb: np.ndarray, count: int
    ) -> np.ndarray:
        """Update centroid/prototype using online mean algorithm."""
        if current is None:
            return new_emb.copy()
        return l2_normalize(current * ((count - 1) / count) + new_emb / count)

    def _create_bidirectional_edge(
        self, from_id: str, to_id: str, forward_type: str, backward_type: str
    ) -> None:
        """Create bidirectional edges in the memory graph."""
        self.graph.ensure_edge(from_id, to_id, forward_type)
        self.graph.ensure_edge(to_id, from_id, backward_type)

    # -------------------------------------------------------------------------
    # VOI feature computation
    # -------------------------------------------------------------------------

    def _compute_redundancy(self, emb: np.ndarray) -> float:
        """Compute redundancy with existing memories."""
        hits = self.vindex.search(emb, k=8, kind="plot")
        return max((s for _, s in hits), default=0.0)

    def _compute_goal_relevance(self, emb: np.ndarray, context_emb: Optional[np.ndarray]) -> float:
        """Compute relevance to current goal/context."""
        return cosine_sim(emb, context_emb) if context_emb is not None else 0.0

    def _compute_pred_error(self, emb: np.ndarray) -> float:
        """Compute predictive error vs best-matching story centroid."""
        best_sim = -1.0
        for story in self.stories.values():
            if story.centroid is None:
                continue
            sim = self.metric.sim(emb, story.centroid)
            if sim > best_sim:
                best_sim = sim
        return 1.0 if best_sim < 0 else 1.0 - best_sim

    def _compute_voi_features(self, plot: Plot) -> np.ndarray:
        """Compute value-of-information features for encoding decision."""
        return np.array([
            plot.surprise,
            plot.pred_error,
            1.0 - plot.redundancy,
            plot.goal_relevance,
            math.tanh(len(plot.text) / TEXT_LENGTH_NORMALIZATION),
            1.0,
        ], dtype=np.float32)

    # -------------------------------------------------------------------------
    # Ingest: Main entry point for new interactions
    # -------------------------------------------------------------------------

    def ingest(
        self,
        interaction_text: str,
        actors: Optional[Sequence[str]] = None,
        context_text: Optional[str] = None,
        event_id: Optional[str] = None,
    ) -> Plot:
        """Ingest an interaction/event with relationship-centric processing.

        This method follows the identity-first paradigm:
        1) Identify the relationship entity
        2) Assess identity relevance (not just information value)
        3) Extract relational context ("who I am in this relationship")
        4) Extract identity impact ("how this affects who I am")
        5) Store organized by relationship
        """
        actors = tuple(actors) if actors else ("user", "agent")
        emb = self.embedder.embed(interaction_text)

        # Prepare plot with relationship-centric processing
        plot = self._prepare_plot(interaction_text, actors, emb, event_id)

        # Update global density regardless of storage decision (calibration)
        self.kde.add(emb)

        # Compute traditional signals
        context_emb = self.embedder.embed(context_text) if context_text else None
        self._compute_plot_signals(plot, emb, context_emb)

        # Make storage decision
        encode = self._compute_storage_decision(plot)

        if encode:
            self._store_plot(plot)
            self._recent_encoded_plot_ids.append(plot.id)
            logger.debug(
                f"Encoded plot {plot.id}, combined_prob={plot._storage_prob:.3f}"
            )
        else:
            logger.debug(
                f"Dropped plot, combined_prob={plot._storage_prob:.3f}"
            )

        # Pressure management
        self._pressure_manage()
        return plot

    def _prepare_plot(
        self,
        interaction_text: str,
        actors: Tuple[str, ...],
        emb: np.ndarray,
        event_id: Optional[str],
    ) -> Plot:
        """Prepare a plot with relationship-centric context."""
        # Relationship identification
        relationship_entity = self._identify_relationship_entity(actors, interaction_text)

        # Identity relevance assessment
        identity_relevance = self._assess_identity_relevance(interaction_text, relationship_entity, emb)

        # Extract relational context
        relational_context = self._extract_relational_context(
            interaction_text, relationship_entity, actors, identity_relevance
        )

        # Extract identity impact
        identity_impact = self._extract_identity_impact(
            interaction_text, relational_context, identity_relevance
        )

        return Plot(
            id=det_id("plot", event_id) if event_id else str(uuid.uuid4()),
            ts=now_ts(),
            text=interaction_text,
            actors=tuple(actors),
            embedding=emb,
            relational=relational_context,
            identity_impact=identity_impact,
        )

    def _compute_plot_signals(
        self, plot: Plot, emb: np.ndarray, context_emb: Optional[np.ndarray]
    ) -> None:
        """Compute traditional signals for a plot."""
        plot.surprise = float(self.kde.surprise(emb))
        plot.pred_error = float(self._compute_pred_error(emb))
        plot.redundancy = float(self._compute_redundancy(emb))
        plot.goal_relevance = float(self._compute_goal_relevance(emb, context_emb))
        plot.tension = plot.surprise * (1.0 + plot.pred_error)

    def _compute_storage_decision(self, plot: Plot) -> bool:
        """Compute whether to store this plot."""
        # Combine traditional VOI with identity relevance
        x = self._compute_voi_features(plot)
        voi_decision = self.gate.prob(x)
        
        # Get identity relevance from relational context
        identity_relevance = self._assess_identity_relevance(
            plot.text, 
            plot.get_relationship_entity() or "user",
            plot.embedding
        )
        
        combined_prob = IDENTITY_RELEVANCE_WEIGHT * identity_relevance + VOI_DECISION_WEIGHT * voi_decision
        plot._storage_prob = combined_prob  # Store for logging
        
        return self.rng.random() < combined_prob

    # -------------------------------------------------------------------------
    # Store plot into story + graph weaving
    # -------------------------------------------------------------------------

    def _store_plot(self, plot: Plot) -> None:
        """Store plot with relationship-first organization."""
        # Assign plot to story
        story, chosen_id = self._assign_plot_to_story(plot)
        
        # Update story with plot
        self._update_story_with_plot(story, plot)
        
        # Store plot and weave edges
        self._weave_plot_edges(plot, story)
        
        # Update identity dimensions
        self._update_identity_dimensions(plot)

    def _assign_plot_to_story(self, plot: Plot) -> Tuple[StoryArc, str]:
        """Assign a plot to an existing or new story."""
        relationship_entity = plot.get_relationship_entity()
        
        if relationship_entity:
            # Relationship-first: get or create story for this relationship
            story = self._get_or_create_relationship_story(relationship_entity)
            chosen_id = story.id
            
            # Add to vector index if this is a new story
            if story.centroid is None:
                self.vindex.add(story.id, plot.embedding, kind="story")
        else:
            # Fallback to CRP for non-relational plots
            logps: Dict[str, float] = {}
            for sid, story in self.stories.items():
                prior = math.log(len(story.plot_ids) + EPSILON_PRIOR)
                logps[sid] = prior + self.story_model.loglik(plot, story)

            chosen_id, _ = self.crp_story.sample(logps)
            if chosen_id is None:
                story = StoryArc(id=det_id("story", plot.id), created_ts=now_ts(), updated_ts=now_ts())
                self.stories[story.id] = story
                self.graph.add_node(story.id, "story", story)
                self.vindex.add(story.id, plot.embedding, kind="story")
                chosen_id = story.id
            else:
                story = self.stories[chosen_id]

        return story, chosen_id

    def _update_story_with_plot(self, story: StoryArc, plot: Plot) -> None:
        """Update story statistics and centroid with a new plot."""
        # Update statistics
        if story.centroid is not None:
            d2 = self.metric.d2(plot.embedding, story.centroid)
            story._update_stats("dist", d2)
            gap = max(0.0, plot.ts - story.updated_ts)
            story._update_stats("gap", gap)

        story.plot_ids.append(plot.id)
        story.updated_ts = plot.ts
        story.actor_counts = {a: story.actor_counts.get(a, 0) + 1 for a in plot.actors}
        story.tension_curve.append(plot.tension)

        # Update centroid
        story.centroid = self._update_centroid_online(
            story.centroid, plot.embedding, len(story.plot_ids)
        )

        # Update relationship trajectory
        if plot.relational and story.is_relationship_story():
            self._update_relationship_trajectory(story, plot)

    def _update_relationship_trajectory(self, story: StoryArc, plot: Plot) -> None:
        """Update relationship trajectory with a new plot."""
        story.add_relationship_moment(
            event_summary=plot.text[:EVENT_SUMMARY_MAX_LENGTH] + "..." if len(plot.text) > EVENT_SUMMARY_MAX_LENGTH else plot.text,
            trust_level=TRUST_BASE + plot.relational.relationship_quality_delta,
            my_role=plot.relational.my_role_in_relation,
            quality_delta=plot.relational.relationship_quality_delta,
            ts=plot.ts,
        )
        
        # Update identity based on accumulated evidence
        if len(story.relationship_arc) >= 3:
            recent_roles = [m.my_role for m in story.relationship_arc[-10:]]
            role_counts = Counter(recent_roles)
            dominant_role = role_counts.most_common(1)[0][0] if role_counts else "助手"
            story.update_identity_in_relationship(dominant_role)

    def _weave_plot_edges(self, plot: Plot, story: StoryArc) -> None:
        """Store plot and weave edges in the graph."""
        plot.story_id = story.id
        self.plots[plot.id] = plot
        self.graph.add_node(plot.id, "plot", plot)
        self.vindex.add(plot.id, plot.embedding, kind="plot")

        # Bidirectional edge to story
        self._create_bidirectional_edge(plot.id, story.id, "belongs_to", "contains")

        # Temporal edge to previous plot in story
        if len(story.plot_ids) > 1:
            prev_id = story.plot_ids[-2]
            self.graph.ensure_edge(prev_id, plot.id, "temporal")

        # Semantic edges to nearest neighbors
        for pid, _ in self.vindex.search(plot.embedding, k=SEMANTIC_NEIGHBORS_K, kind="plot"):
            if pid != plot.id:
                self.graph.ensure_edge(plot.id, pid, "semantic")
                self.graph.ensure_edge(pid, plot.id, "semantic")

    # -------------------------------------------------------------------------
    # Query / retrieval
    # -------------------------------------------------------------------------

    def query(self, text: str, k: int = 5, asker_id: Optional[str] = None) -> RetrievalTrace:
        """Query with relationship-first retrieval."""
        # Relationship identification and identity activation
        activated_identity = None
        relationship_context = None
        relationship_story = None
        
        if asker_id:
            story_id = self._relationship_story_index.get(asker_id)
            if story_id:
                relationship_story = self.stories.get(story_id)
                if relationship_story:
                    activated_identity = relationship_story.my_identity_in_this_relationship
                    relationship_context = relationship_story.to_relationship_narrative()
        
        # Retrieve results
        if relationship_story and activated_identity:
            trace = self._retrieve_with_relationship_priority(text, relationship_story, k=k)
        else:
            trace = self.retriever.retrieve(
                query_text=text, embed=self.embedder, 
                kinds=self.cfg.retrieval_kinds, k=k
            )
        
        # Enrich trace with relationship context
        trace.asker_id = asker_id
        trace.activated_identity = activated_identity
        trace.relationship_context = relationship_context

        # Update access/mass
        self._update_access_counts(trace)
        
        return trace

    def _retrieve_with_relationship_priority(
        self, text: str, relationship_story: StoryArc, k: int
    ) -> RetrievalTrace:
        """Retrieve with priority given to the relationship's history."""
        query_emb = self.embedder.embed(text)
        
        # Get relationship-specific results
        relationship_results = self._get_relationship_results(query_emb, relationship_story)
        
        # Get semantic results from other memories
        semantic_trace = self.retriever.retrieve(
            query_text=text, embed=self.embedder,
            kinds=self.cfg.retrieval_kinds, k=k
        )
        
        # Merge results
        ranked = self._merge_retrieval_results(relationship_results, semantic_trace.ranked, k)
        
        return RetrievalTrace(
            query=text,
            query_emb=query_emb,
            attractor_path=semantic_trace.attractor_path,
            ranked=ranked,
        )

    def _get_relationship_results(
        self, query_emb: np.ndarray, relationship_story: StoryArc
    ) -> List[Tuple[str, float, str]]:
        """Get retrieval results from a relationship's history."""
        results: List[Tuple[str, float, str]] = []
        
        # Score plots in this relationship's story
        for plot_id in relationship_story.plot_ids[-MAX_RECENT_PLOTS_FOR_RETRIEVAL:]:
            plot = self.plots.get(plot_id)
            if plot is None or plot.status != "active":
                continue
            
            sem_sim = self.metric.sim(query_emb, plot.embedding)
            score = sem_sim + RELATIONSHIP_BONUS_SCORE
            results.append((plot_id, score, "plot"))
        
        # Add the relationship story itself
        if relationship_story.centroid is not None:
            story_sim = self.metric.sim(query_emb, relationship_story.centroid)
            results.append((relationship_story.id, story_sim + STORY_SIMILARITY_BONUS, "story"))
        
        return results

    def _merge_retrieval_results(
        self,
        relationship_results: List[Tuple[str, float, str]],
        semantic_results: List[Tuple[str, float, str]],
        k: int,
    ) -> List[Tuple[str, float, str]]:
        """Merge relationship and semantic retrieval results."""
        all_results: Dict[str, Tuple[float, str]] = {}
        
        # Add relationship results (higher priority)
        for nid, score, kind in relationship_results:
            all_results[nid] = (score, kind)
        
        # Add semantic results (don't override if already present with higher score)
        for nid, score, kind in semantic_results:
            if nid not in all_results:
                all_results[nid] = (score, kind)
            else:
                existing_score, existing_kind = all_results[nid]
                if score > existing_score:
                    all_results[nid] = (score, kind)
        
        # Sort and take top k
        return sorted(
            [(nid, score, kind) for nid, (score, kind) in all_results.items()],
            key=lambda x: x[1],
            reverse=True
        )[:k]

    def _update_access_counts(self, trace: RetrievalTrace) -> None:
        """Update access counts for retrieved items."""
        for nid, _, kind in trace.ranked:
            if kind == "plot":
                plot = self.graph.payload(nid)
                plot.access_count += 1
                plot.last_access_ts = now_ts()
            elif kind == "story":
                story = self.graph.payload(nid)
                story.reference_count += 1

    # -------------------------------------------------------------------------
    # Feedback: credit assignment and learning
    # -------------------------------------------------------------------------

    def feedback_retrieval(self, query_text: str, chosen_id: str, success: bool) -> None:
        """Delayed reward signal for learning."""
        import networkx as nx
        
        query_emb = self.embedder.embed(query_text)
        graph = self.graph.g

        # Update edges on shortest paths from seeds
        self._update_edge_beliefs(query_emb, chosen_id, success, graph)

        # Metric triplet update
        self._update_metric_triplet(query_emb, chosen_id, success, graph)

        # Encode gate update
        self._update_encode_gate(success)

        # Theme evidence update
        if chosen_id in self.themes:
            self.themes[chosen_id].update_evidence(success)

    def _update_edge_beliefs(
        self, query_emb: np.ndarray, chosen_id: str, success: bool, graph
    ) -> None:
        """Update edge beliefs on shortest paths."""
        import networkx as nx
        
        seeds = [i for i, _ in self.vindex.search(query_emb, k=10)]
        if chosen_id in graph:
            for seed in seeds:
                if seed not in graph:
                    continue
                try:
                    path = nx.shortest_path(graph, source=seed, target=chosen_id)
                except nx.NetworkXNoPath:
                    continue
                for u, v in zip(path[:-1], path[1:]):
                    self.graph.edge_belief(u, v).update(success)

    def _update_metric_triplet(
        self, query_emb: np.ndarray, chosen_id: str, success: bool, graph
    ) -> None:
        """Update metric using triplet loss."""
        if chosen_id not in graph:
            return
            
        chosen = self.graph.payload(chosen_id)
        pos_emb = getattr(chosen, "embedding", getattr(chosen, "centroid", getattr(chosen, "prototype", None)))
        
        if pos_emb is None:
            return
            
        # Pick negative among high-sim but not chosen
        cands = [i for i, _ in self.vindex.search(query_emb, k=30) if i != chosen_id and i in graph]
        if not cands:
            return
            
        neg_id = self.rng.choice(cands)
        neg = self.graph.payload(neg_id)
        neg_emb = getattr(neg, "embedding", getattr(neg, "centroid", getattr(neg, "prototype", None)))
        
        if neg_emb is not None:
            self.metric.update_triplet(anchor=query_emb, positive=pos_emb, negative=neg_emb)

    def _update_encode_gate(self, success: bool) -> None:
        """Update encode gate with reward signal."""
        reward = 1.0 if success else -1.0
        recent = list(self._recent_encoded_plot_ids)[-RECENT_PLOTS_FOR_FEEDBACK:]
        
        for pid in recent:
            plot = self.plots.get(pid)
            if plot is not None:
                x = self._compute_voi_features(plot)
                self.gate.update(x, reward)

    # -------------------------------------------------------------------------
    # Evolution: consolidation & theme emergence
    # -------------------------------------------------------------------------

    def evolve(self) -> None:
        """Offline-ish evolution step: "持续成为" (continuous becoming)."""
        logger.info(
            f"Starting evolution: plots={len(self.plots)}, "
            f"stories={len(self.stories)}, themes={len(self.themes)}"
        )
        
        # Relationship Reflection
        self._reflect_on_relationships()
        
        # Meaning Reframe Check
        self._check_reframe_opportunities()

        # Story Boundary Detection (climax, resolution, abandonment)
        self._detect_story_boundaries()

        # Update story statuses (probabilistic)
        self._update_story_statuses()

        # Theme/Identity Dimension Emergence
        self._process_theme_emergence()

        # Identity Dimension Tension Analysis
        self._analyze_identity_tensions()

        # Graph Structure Cleanup (weak edges, similar nodes, stale content)
        self._cleanup_graph_structure()

        # Growth-oriented pressure management
        self._pressure_manage()
        
        logger.info(
            f"Evolution complete: plots={len(self.plots)}, "
            f"stories={len(self.stories)}, themes={len(self.themes)}"
        )

    def _update_story_statuses(self) -> None:
        """Update story statuses based on activity probability."""
        for story in self.stories.values():
            if story.status != "developing":
                continue
            
            p_active = story.activity_probability()
            if self.rng.random() < p_active:
                continue
            
            # Resolve vs abandon
            if len(story.tension_curve) >= 3:
                slope = story.tension_curve[-1] - story.tension_curve[0]
                p_resolve = sigmoid(-slope)
            else:
                p_resolve = 0.5
            
            story.status = "resolved" if self.rng.random() < p_resolve else "abandoned"

    def _process_theme_emergence(self) -> None:
        """Process theme emergence from resolved stories."""
        for sid, story in list(self.stories.items()):
            if story.status != "resolved" or story.centroid is None:
                continue
            
            # Compute log probabilities for existing themes
            logps: Dict[str, float] = {}
            for tid, theme in self.themes.items():
                prior = math.log(len(theme.story_ids) + EPSILON_PRIOR)
                logps[tid] = prior + self.theme_model.loglik(story, theme)
            
            chosen_id, _ = self.crp_theme.sample(logps)
            
            if chosen_id is None:
                # Create new theme
                theme = Theme(id=det_id("theme", sid), created_ts=now_ts(), updated_ts=now_ts())
                theme.prototype = story.centroid.copy()
                
                # Set identity dimension from relationship story
                if story.is_relationship_story() and story.my_identity_in_this_relationship:
                    theme.identity_dimension = f"作为{story.my_identity_in_this_relationship}的我"
                    theme.theme_type = "identity"
                    if story.relationship_with:
                        theme.supporting_relationships.append(story.relationship_with)
                
                self.themes[theme.id] = theme
                self.graph.add_node(theme.id, "theme", theme)
                self.vindex.add(theme.id, theme.prototype, kind="theme")
                chosen_id = theme.id
            
            theme = self.themes[chosen_id]
            theme.story_ids.append(sid)
            theme.updated_ts = now_ts()
            
            # Update identity dimension support
            if story.is_relationship_story() and story.relationship_with:
                theme.add_supporting_relationship(story.relationship_with)
            
            # Update prototype
            theme.prototype = self._update_centroid_online(
                theme.prototype, story.centroid, len(theme.story_ids)
            )

            # Weave edges
            self._create_bidirectional_edge(sid, theme.id, "thematizes", "exemplified_by")

    # -------------------------------------------------------------------------
    # Convenience: inspect
    # -------------------------------------------------------------------------

    def get_story(self, story_id: str) -> StoryArc:
        """Get a story by ID."""
        return self.stories[story_id]

    def get_plot(self, plot_id: str) -> Plot:
        """Get a plot by ID."""
        return self.plots[plot_id]

    def get_theme(self, theme_id: str) -> Theme:
        """Get a theme by ID."""
        return self.themes[theme_id]
    
    def get_relationship_story(self, entity_id: str) -> Optional[StoryArc]:
        """Get the story for a specific relationship entity."""
        story_id = self._relationship_story_index.get(entity_id)
        return self.stories.get(story_id) if story_id else None
    
    def get_my_identity_with(self, entity_id: str) -> Optional[str]:
        """Get my identity in a specific relationship."""
        story = self.get_relationship_story(entity_id)
        return story.my_identity_in_this_relationship if story else None
    
    def get_all_relationships(self) -> Dict[str, StoryArc]:
        """Get all relationship stories."""
        return {
            entity: self.stories[story_id]
            for entity, story_id in self._relationship_story_index.items()
            if story_id in self.stories
        }
    
    def get_identity_summary(self) -> Dict[str, Any]:
        """Get a summary of the agent's identity across all relationships."""
        relationships = self.get_all_relationships()
        
        summary = {
            "identity_dimensions": dict(self._identity_dimensions),
            "relationship_identities": {
                entity: story.my_identity_in_this_relationship
                for entity, story in relationships.items()
            },
            "relationship_count": len(relationships),
            "total_interactions": sum(len(story.plot_ids) for story in relationships.values()),
        }
        
        if self._identity_dimensions:
            dominant = max(self._identity_dimensions, key=self._identity_dimensions.get)
            summary["dominant_dimension"] = dominant
        
        return summary


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
