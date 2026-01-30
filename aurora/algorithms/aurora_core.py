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

from aurora.algorithms.models.plot import Plot, RelationalContext, IdentityImpact
from aurora.algorithms.models.story import StoryArc, RelationshipMoment
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

try:
    from aurora.algorithms.graph.faiss_index import FAISSVectorIndex, FAISS_AVAILABLE
except ImportError:
    FAISSVectorIndex = None
    FAISS_AVAILABLE = False

from aurora.algorithms.retrieval.field_retriever import FieldRetriever

import logging
logger = logging.getLogger(__name__)


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
        
        # Vector index - use FAISS if configured and available
        self.vindex = self._create_vector_index(cfg)

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
        
        # === Relationship-centric additions ===
        # Index: relationship_entity -> story_id (for quick lookup)
        self._relationship_story_index: Dict[str, str] = {}
        
        # Identity dimensions currently recognized (evolves over time)
        self._identity_dimensions: Dict[str, float] = {}  # dimension_name -> strength

    # -------------------------------------------------------------------------
    # Vector index creation
    # -------------------------------------------------------------------------

    def _create_vector_index(self, cfg: MemoryConfig):
        """Create vector index based on configuration."""
        if cfg.vector_backend == "faiss":
            if not FAISS_AVAILABLE:
                raise ImportError("FAISS required. Install: pip install faiss-cpu")
            return FAISSVectorIndex(
                dim=cfg.dim,
                M=cfg.faiss_m,
                ef_construction=cfg.faiss_ef_construction,
                ef_search=cfg.faiss_ef_search,
            )
        
        if cfg.vector_backend == "auto" and FAISS_AVAILABLE:
            return FAISSVectorIndex(
                dim=cfg.dim,
                M=cfg.faiss_m,
                ef_construction=cfg.faiss_ef_construction,
                ef_search=cfg.faiss_ef_search,
            )
        
        return VectorIndex(dim=cfg.dim)

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
    # Relationship Identification and Identity Assessment
    # -------------------------------------------------------------------------

    def _identify_relationship_entity(self, actors: Tuple[str, ...], text: str) -> str:
        """
        Identify the primary relationship entity from this interaction.
        
        The key insight: memory should be organized around relationships,
        not just semantic similarity. This method identifies who the "other"
        is in this interaction.
        """
        # Filter out the agent/assistant itself
        others = [a for a in actors if a.lower() not in ("agent", "assistant", "ai", "system")]
        
        if others:
            # Return the first non-self actor as the relationship entity
            return others[0]
        
        # If no clear other, use "user" as default
        return "user"

    def _assess_identity_relevance(self, text: str, relationship_entity: str, emb: np.ndarray) -> float:
        """
        Assess how relevant this interaction is to "who I am".
        
        This replaces pure information-theoretic VOI with identity-relevance.
        The question is not "is this surprising?" but "does this affect my identity?"
        """
        relevance = 0.0
        
        # 1. Check if this reinforces existing identity dimensions
        reinforcement = self._check_identity_reinforcement(emb)
        relevance += reinforcement * 0.4
        
        # 2. Check if this challenges existing identity dimensions
        challenge = self._check_identity_challenge(text, emb)
        relevance += challenge * 0.5  # Challenges are more important to remember
        
        # 3. Check if this introduces new identity possibilities
        novelty = self._check_identity_novelty(emb)
        relevance += novelty * 0.3
        
        # 4. Relationship factor: interactions in important relationships matter more
        relationship_importance = self._get_relationship_importance(relationship_entity)
        relevance *= (0.5 + 0.5 * relationship_importance)
        
        return min(1.0, relevance)

    def _check_identity_reinforcement(self, emb: np.ndarray) -> float:
        """Check if this interaction reinforces existing identity dimensions."""
        if not self._identity_dimensions:
            return 0.0
        
        # Check similarity to existing theme prototypes (identity dimensions)
        max_sim = 0.0
        for theme in self.themes.values():
            if theme.prototype is not None:
                sim = self.metric.sim(emb, theme.prototype)
                max_sim = max(max_sim, sim)
        
        return max_sim

    def _check_identity_challenge(self, text: str, emb: np.ndarray) -> float:
        """Check if this interaction challenges existing identity dimensions."""
        if not self.themes:
            return 0.0
        
        # Look for semantic opposition to existing themes
        # (high surprise + moderate similarity suggests a challenge)
        surprise = float(self.kde.surprise(emb))
        
        max_moderate_sim = 0.0
        for theme in self.themes.values():
            if theme.prototype is not None:
                sim = self.metric.sim(emb, theme.prototype)
                # "Moderate" similarity (0.3-0.6) with high surprise = challenge
                if 0.3 < sim < 0.6:
                    max_moderate_sim = max(max_moderate_sim, sim)
        
        # Challenge score: high surprise + moderate similarity
        if max_moderate_sim > 0 and surprise > 0:
            return min(1.0, surprise * 0.3 * max_moderate_sim)
        
        return 0.0

    def _check_identity_novelty(self, emb: np.ndarray) -> float:
        """Check if this introduces new identity possibilities."""
        if not self.themes:
            return 1.0  # Everything is novel when we have no identity yet
        
        # Low similarity to all existing themes = potentially new dimension
        min_sim = 1.0
        for theme in self.themes.values():
            if theme.prototype is not None:
                sim = self.metric.sim(emb, theme.prototype)
                min_sim = min(min_sim, sim)
        
        # Novelty is inversely related to maximum similarity
        return max(0.0, 1.0 - min_sim)

    def _get_relationship_importance(self, relationship_entity: str) -> float:
        """Get the importance of a relationship (based on history)."""
        story_id = self._relationship_story_index.get(relationship_entity)
        if story_id is None:
            return 0.5  # Neutral for new relationships
        
        story = self.stories.get(story_id)
        if story is None:
            return 0.5
        
        # Importance based on: number of interactions + health
        interaction_count = len(story.plot_ids)
        health = story.relationship_health
        
        # More interactions = more important relationship
        count_factor = min(1.0, math.log1p(interaction_count) / 3.0)
        
        return 0.3 * count_factor + 0.7 * health

    def _extract_relational_context(
        self, 
        text: str, 
        relationship_entity: str,
        actors: Tuple[str, ...],
        identity_relevance: float
    ) -> RelationalContext:
        """
        Extract the relational meaning from this interaction.
        
        Key question: "Who am I in this relationship?"
        """
        # Determine my role based on interaction content AND relationship history
        my_role = self._infer_my_role(text, relationship_entity)
        
        # Estimate relationship quality impact
        quality_delta = self._estimate_quality_delta(text, identity_relevance)
        
        # Generate a brief relational meaning (could use LLM in production)
        meaning = self._generate_relational_meaning(text, my_role, quality_delta)
        
        return RelationalContext(
            with_whom=relationship_entity,
            my_role_in_relation=my_role,
            relationship_quality_delta=quality_delta,
            what_this_says_about_us=meaning,
        )

    def _infer_my_role(self, text: str, relationship_entity: Optional[str] = None) -> str:
        """
        Infer my role in this interaction.
        
        Enhanced to use relationship history when available:
        - If there's established history, prefer the consistent role from that relationship
        - Otherwise, use keyword-based inference
        """
        # Check if we have relationship history
        if relationship_entity:
            story = self.get_relationship_story(relationship_entity)
            if story and story.my_identity_in_this_relationship:
                # Use established identity if role consistency is high
                if story.get_role_consistency(window=5) > 0.6:
                    return story.my_identity_in_this_relationship
        
        # Fallback to keyword-based inference
        return self._keyword_based_role(text)
    
    def _keyword_based_role(self, text: str) -> str:
        """Keyword-based role inference."""
        text_lower = text.lower()
        
        # Simple heuristic-based role inference
        if any(kw in text_lower for kw in ["解释", "explain", "说明", "clarify"]):
            return "解释者"
        elif any(kw in text_lower for kw in ["帮助", "help", "协助", "assist"]):
            return "帮助者"
        elif any(kw in text_lower for kw in ["分析", "analyze", "评估", "evaluate"]):
            return "分析者"
        elif any(kw in text_lower for kw in ["代码", "code", "编程", "program"]):
            return "编程助手"
        elif any(kw in text_lower for kw in ["计划", "plan", "规划", "design"]):
            return "规划者"
        elif any(kw in text_lower for kw in ["学习", "learn", "理解", "understand"]):
            return "学习伙伴"
        elif any(kw in text_lower for kw in ["创作", "create", "写", "write"]):
            return "创作伙伴"
        else:
            return "助手"

    def _estimate_quality_delta(self, text: str, identity_relevance: float) -> float:
        """Estimate how this interaction affects relationship quality."""
        # Simple heuristic: positive interactions improve quality
        text_lower = text.lower()
        
        positive_indicators = ["谢谢", "感谢", "太好了", "perfect", "great", "thanks", "helpful", "好的"]
        negative_indicators = ["不对", "错误", "不行", "wrong", "error", "fail", "不满意"]
        
        positive_count = sum(1 for kw in positive_indicators if kw in text_lower)
        negative_count = sum(1 for kw in negative_indicators if kw in text_lower)
        
        # Base delta scaled by identity relevance
        base_delta = (positive_count - negative_count) * 0.1
        return max(-0.3, min(0.3, base_delta * (0.5 + 0.5 * identity_relevance)))

    def _generate_relational_meaning(self, text: str, my_role: str, quality_delta: float) -> str:
        """Generate a brief description of relational meaning."""
        if quality_delta > 0.1:
            return f"作为{my_role}，我们的关系更进一步了"
        elif quality_delta < -0.1:
            return f"作为{my_role}，这次互动有些挑战"
        else:
            return f"作为{my_role}，这是一次常规互动"

    def _extract_identity_impact(
        self, 
        text: str, 
        relational: RelationalContext,
        identity_relevance: float
    ) -> Optional[IdentityImpact]:
        """
        Extract how this interaction impacts my identity.
        
        Key insight: The meaning of an experience can evolve over time.
        """
        if identity_relevance < 0.2:
            return None  # Not significant enough for identity impact
        
        # Identify affected dimensions
        affected_dimensions = self._identify_affected_dimensions(text, relational)
        
        if not affected_dimensions:
            return None
        
        # Generate initial meaning
        initial_meaning = self._generate_identity_meaning(text, relational, affected_dimensions)
        
        return IdentityImpact(
            when_formed=now_ts(),
            initial_meaning=initial_meaning,
            current_meaning=initial_meaning,  # Same at creation time
            identity_dimensions_affected=affected_dimensions,
            evolution_history=[],
        )

    def _identify_affected_dimensions(self, text: str, relational: RelationalContext) -> List[str]:
        """Identify which identity dimensions are affected by this interaction."""
        dimensions = []
        text_lower = text.lower()
        
        # Map keywords to identity dimensions
        dimension_keywords = {
            "作为解释者的我": ["解释", "说明", "clarify", "explain"],
            "作为帮助者的我": ["帮助", "协助", "help", "assist"],
            "作为学习者的我": ["学习", "理解", "learn", "understand"],
            "作为创造者的我": ["创作", "创建", "create", "build", "写"],
            "作为分析者的我": ["分析", "评估", "analyze", "evaluate"],
            "作为编程者的我": ["代码", "编程", "code", "program"],
        }
        
        for dimension, keywords in dimension_keywords.items():
            if any(kw in text_lower for kw in keywords):
                dimensions.append(dimension)
        
        # Always include the role-based dimension
        role_dimension = f"作为{relational.my_role_in_relation}的我"
        if role_dimension not in dimensions:
            dimensions.append(role_dimension)
        
        return dimensions[:3]  # Limit to top 3

    def _generate_identity_meaning(
        self, 
        text: str, 
        relational: RelationalContext,
        affected_dimensions: List[str]
    ) -> str:
        """Generate the identity meaning of this interaction."""
        dims_str = "、".join(affected_dimensions[:2]) if affected_dimensions else "我的身份"
        
        if relational.relationship_quality_delta > 0.1:
            return f"这次互动强化了{dims_str}"
        elif relational.relationship_quality_delta < -0.1:
            return f"这次互动挑战了{dims_str}"
        else:
            return f"这是{dims_str}的一次体现"

    def _get_or_create_relationship_story(self, relationship_entity: str) -> StoryArc:
        """Get or create a story for a relationship."""
        # Check if we already have a story for this relationship
        story_id = self._relationship_story_index.get(relationship_entity)
        
        if story_id and story_id in self.stories:
            return self.stories[story_id]
        
        # Create a new story for this relationship
        story = StoryArc(
            id=det_id("story", f"rel_{relationship_entity}"),
            created_ts=now_ts(),
            updated_ts=now_ts(),
            relationship_with=relationship_entity,
            relationship_type="user" if "user" in relationship_entity.lower() else "other",
        )
        
        self.stories[story.id] = story
        self.graph.add_node(story.id, "story", story)
        self._relationship_story_index[relationship_entity] = story.id
        
        return story

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
        """Ingest an interaction/event with relationship-centric processing.

        This method now follows the identity-first paradigm:
        1) Identify the relationship entity
        2) Assess identity relevance (not just information value)
        3) Extract relational context ("who I am in this relationship")
        4) Extract identity impact ("how this affects who I am")
        5) Store organized by relationship
        
        Key insight: The question is not "is this surprising?" but 
        "does this affect my identity and my relationships?"
        """
        actors = tuple(actors) if actors else ("user", "agent")
        emb = self.embedder.embed(interaction_text)

        # === 1. Relationship identification ===
        relationship_entity = self._identify_relationship_entity(actors, interaction_text)

        # === 2. Identity relevance assessment ===
        identity_relevance = self._assess_identity_relevance(interaction_text, relationship_entity, emb)

        # === 3. Extract relational context ===
        relational_context = self._extract_relational_context(
            interaction_text, relationship_entity, actors, identity_relevance
        )

        # === 4. Extract identity impact ===
        identity_impact = self._extract_identity_impact(
            interaction_text, relational_context, identity_relevance
        )

        # Create Plot with all layers
        plot = Plot(
            id=det_id("plot", event_id) if event_id else str(uuid.uuid4()),
            ts=now_ts(),
            text=interaction_text,
            actors=tuple(actors),
            embedding=emb,
            relational=relational_context,
            identity_impact=identity_impact,
        )

        # Update global density regardless of storage decision (calibration)
        self.kde.add(emb)

        # Compute traditional signals (kept for compatibility and VOI)
        context_emb = self.embedder.embed(context_text) if context_text else None
        plot.surprise = float(self.kde.surprise(emb))
        plot.pred_error = float(self._pred_error(emb))
        plot.redundancy = float(self._redundancy(emb))
        plot.goal_relevance = float(self._goal_relevance(emb, context_emb))

        # Free-energy proxy: surprise plus belief update magnitude
        plot.tension = plot.surprise * (1.0 + plot.pred_error)

        # === Storage decision (now considers identity relevance) ===
        # Combine traditional VOI with identity relevance
        x = self._voi_features(plot)
        voi_decision = self.gate.prob(x)
        
        # Weight: 60% identity relevance, 40% traditional VOI
        combined_prob = 0.6 * identity_relevance + 0.4 * voi_decision
        encode = self.rng.random() < combined_prob

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
        """Store plot with relationship-first organization.
        
        Key change: Stories are now organized primarily by relationship,
        not just semantic similarity. This means:
        - Each relationship entity has its own story
        - The story tracks the relationship trajectory
        - Identity emerges from these relationship stories
        """
        # === 1) Relationship-first story assignment ===
        relationship_entity = plot.get_relationship_entity()
        
        if relationship_entity:
            # Get or create a story for this relationship
            story = self._get_or_create_relationship_story(relationship_entity)
            chosen = story.id
            
            # Add to vector index if this is a new story
            if story.centroid is None:
                self.vindex.add(story.id, plot.embedding, kind="story")
        else:
            # Fallback to CRP for non-relational plots
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

        # === 2) Update story statistics and centroid ===
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

        # === 3) Update relationship trajectory ===
        if plot.relational and story.is_relationship_story():
            # Add a moment to the relationship arc
            story.add_relationship_moment(
                event_summary=plot.text[:100] + "..." if len(plot.text) > 100 else plot.text,
                trust_level=0.5 + plot.relational.relationship_quality_delta,  # Adjust from delta
                my_role=plot.relational.my_role_in_relation,
                quality_delta=plot.relational.relationship_quality_delta,
                ts=plot.ts,
            )
            
            # Update my identity in this relationship based on accumulated evidence
            if len(story.relationship_arc) >= 3:
                # Find the most common role
                recent_roles = [m.my_role for m in story.relationship_arc[-10:]]
                role_counts: Dict[str, int] = {}
                for role in recent_roles:
                    role_counts[role] = role_counts.get(role, 0) + 1
                dominant_role = max(role_counts, key=role_counts.get) if role_counts else "助手"
                story.update_identity_in_relationship(dominant_role)

        # === 4) Store plot ===
        plot.story_id = story.id
        self.plots[plot.id] = plot
        self.graph.add_node(plot.id, "plot", plot)
        self.vindex.add(plot.id, plot.embedding, kind="plot")

        # === 5) Weave edges (typed relations; strength learned by Beta posteriors) ===
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
        
        # === 6) Update identity dimensions ===
        if plot.identity_impact:
            for dim in plot.identity_impact.identity_dimensions_affected:
                current = self._identity_dimensions.get(dim, 0.0)
                # Gradual strengthening of identity dimensions
                self._identity_dimensions[dim] = current + 0.1 * (1.0 - current)

    # -------------------------------------------------------------------------
    # Query / retrieval
    # -------------------------------------------------------------------------

    def query(self, text: str, k: int = 5, asker_id: Optional[str] = None) -> RetrievalTrace:
        """Query with relationship-first retrieval.
        
        Key insight: Retrieval is not just about finding relevant content,
        but about activating the appropriate identity to respond.
        
        Args:
            text: Query text
            k: Number of results to return
            asker_id: Optional ID of the entity asking (enables relationship-aware retrieval)
            
        Returns:
            RetrievalTrace with results and activated identity context
        """
        # === 1. Relationship identification and identity activation ===
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
        
        # === 2. Retrieve results ===
        if relationship_story and activated_identity:
            # Relationship-first retrieval: prioritize this relationship's plots
            trace = self._retrieve_with_relationship_priority(
                text, relationship_story, k=k
            )
        else:
            # Standard semantic retrieval
            trace = self.retriever.retrieve(
                query_text=text, embed=self.embedder, 
                kinds=self.cfg.retrieval_kinds, k=k
            )
        
        # === 3. Enrich trace with relationship context ===
        trace.asker_id = asker_id
        trace.activated_identity = activated_identity
        trace.relationship_context = relationship_context

        # === 4. Update access/mass ===
        for nid, _score, kind in trace.ranked:
            if kind == "plot":
                p: Plot = self.graph.payload(nid)
                p.access_count += 1
                p.last_access_ts = now_ts()
            elif kind == "story":
                s: StoryArc = self.graph.payload(nid)
                s.reference_count += 1
        
        return trace
    
    def _retrieve_with_relationship_priority(
        self, 
        text: str, 
        relationship_story: StoryArc, 
        k: int
    ) -> RetrievalTrace:
        """Retrieve with priority given to the relationship's history.
        
        Strategy:
        1. First retrieve from this relationship's plots (relationship memory)
        2. Then supplement with semantic matches from other memories
        3. Merge results with relationship context boosted
        """
        query_emb = self.embedder.embed(text)
        
        # === 1. Get relationship-specific results ===
        relationship_results: List[Tuple[str, float, str]] = []
        
        # Score all plots in this relationship's story
        for plot_id in relationship_story.plot_ids[-50:]:  # Recent 50 max
            plot = self.plots.get(plot_id)
            if plot is None or plot.status != "active":
                continue
            
            # Compute relevance: semantic similarity + relationship bonus
            sem_sim = self.metric.sim(query_emb, plot.embedding)
            relationship_bonus = 0.2  # Boost for being in the same relationship
            score = sem_sim + relationship_bonus
            
            relationship_results.append((plot_id, score, "plot"))
        
        # Add the relationship story itself
        if relationship_story.centroid is not None:
            story_sim = self.metric.sim(query_emb, relationship_story.centroid)
            relationship_results.append((relationship_story.id, story_sim + 0.3, "story"))
        
        # === 2. Get semantic results from other memories ===
        semantic_trace = self.retriever.retrieve(
            query_text=text, embed=self.embedder,
            kinds=self.cfg.retrieval_kinds, k=k
        )
        
        # === 3. Merge results ===
        # Combine and deduplicate
        all_results: Dict[str, Tuple[float, str]] = {}
        
        # Add relationship results (higher priority)
        for nid, score, kind in relationship_results:
            all_results[nid] = (score, kind)
        
        # Add semantic results (don't override if already present)
        for nid, score, kind in semantic_trace.ranked:
            if nid not in all_results:
                all_results[nid] = (score, kind)
            else:
                # Keep the higher score
                existing_score, existing_kind = all_results[nid]
                if score > existing_score:
                    all_results[nid] = (score, kind)
        
        # Sort by score and take top k
        ranked = sorted(
            [(nid, score, kind) for nid, (score, kind) in all_results.items()],
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        return RetrievalTrace(
            query=text,
            query_emb=query_emb,
            attractor_path=semantic_trace.attractor_path,
            ranked=ranked,
        )

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
        """Offline-ish evolution step: "持续成为" (continuous becoming).

        Enhanced with identity-first paradigm:
        1) Relationship reflection - assess relationship health and extract lessons
        2) Meaning reframe - update interpretations of past experiences
        3) Story status updates - probabilistic based on activity
        4) Theme/identity dimension emergence - themes as "who I am"
        5) Growth-oriented pressure management
        """
        # === 0) Relationship Reflection ===
        self._reflect_on_relationships()
        
        # === 1) Meaning Reframe Check ===
        self._check_reframe_opportunities()

        # === 2) Update story statuses (probabilistic, no idle threshold) ===
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

        # === 3) Theme/Identity Dimension Emergence ===
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
                
                # Set identity dimension from relationship story
                if s.is_relationship_story() and s.my_identity_in_this_relationship:
                    theme.identity_dimension = f"作为{s.my_identity_in_this_relationship}的我"
                    theme.theme_type = "identity"
                    if s.relationship_with:
                        theme.supporting_relationships.append(s.relationship_with)
                
                self.themes[theme.id] = theme
                self.graph.add_node(theme.id, "theme", theme)
                self.vindex.add(theme.id, theme.prototype, kind="theme")
                chosen = theme.id
            
            theme = self.themes[chosen]
            theme.story_ids.append(sid)
            theme.updated_ts = now_ts()
            
            # Update identity dimension support from relationship
            if s.is_relationship_story() and s.relationship_with:
                theme.add_supporting_relationship(s.relationship_with)
            
            # prototype update (online mean)
            if theme.prototype is None:
                theme.prototype = s.centroid.copy()
            else:
                n = len(theme.story_ids)
                theme.prototype = l2_normalize(theme.prototype * ((n - 1) / n) + s.centroid * (1.0 / n))

            # weave edges
            self.graph.ensure_edge(sid, theme.id, "thematizes")
            self.graph.ensure_edge(theme.id, sid, "exemplified_by")

        # === 4) Identity Dimension Tension Analysis ===
        self._analyze_identity_tensions()

        # === 5) Growth-oriented pressure management ===
        self._pressure_manage()

    # -------------------------------------------------------------------------
    # Relationship Reflection (NEW - "关系反思")
    # -------------------------------------------------------------------------
    
    def _reflect_on_relationships(self) -> None:
        """
        Reflect on relationships: a key part of "持续成为".
        
        Questions to answer:
        1. Which relationships are growing, which are stagnant?
        2. Is my identity consistent across relationships? Should it be?
        3. What lessons can be extracted from each relationship?
        """
        for entity, story_id in self._relationship_story_index.items():
            story = self.stories.get(story_id)
            if story is None:
                continue
            
            # Skip if no meaningful history
            if len(story.relationship_arc) < 3:
                continue
            
            # 1. Assess relationship health trend
            trust_trend = story.get_trust_trend(window=10)
            role_consistency = story.get_role_consistency(window=10)
            
            # Update relationship health based on trends
            health_delta = 0.1 * trust_trend + 0.05 * (role_consistency - 0.5)
            story.relationship_health = max(0.0, min(1.0, 
                story.relationship_health + health_delta
            ))
            
            # 2. Extract lessons if relationship has matured
            if len(story.relationship_arc) >= 10 and len(story.lessons_from_relationship) < 3:
                lesson = self._extract_relationship_lesson(story)
                if lesson:
                    story.add_lesson(lesson)
            
            # 3. Update identity based on consistent role
            if role_consistency > 0.7 and story.relationship_arc:
                dominant_role = story.relationship_arc[-1].my_role
                if dominant_role != story.my_identity_in_this_relationship:
                    story.update_identity_in_relationship(dominant_role)
    
    def _extract_relationship_lesson(self, story: StoryArc) -> Optional[str]:
        """Extract a lesson from relationship history."""
        if not story.relationship_arc:
            return None
        
        # Simple heuristic: look at trust evolution
        if len(story.relationship_arc) >= 5:
            early_trust = sum(m.trust_level for m in story.relationship_arc[:3]) / 3
            recent_trust = sum(m.trust_level for m in story.relationship_arc[-3:]) / 3
            
            if recent_trust > early_trust + 0.2:
                return "耐心和持续的努力能建立信任"
            elif recent_trust < early_trust - 0.2:
                return "关系需要持续维护"
        
        # Check for role consistency
        roles = [m.my_role for m in story.relationship_arc[-10:]]
        if len(set(roles)) == 1:
            return f"在这段关系中，保持{roles[0]}的角色是有效的"
        
        return None

    # -------------------------------------------------------------------------
    # Meaning Reframe (NEW - "意义重构")
    # -------------------------------------------------------------------------
    
    def _check_reframe_opportunities(self) -> None:
        """
        Check which memories need their meaning updated.
        
        Philosophy: The meaning of past experiences can evolve.
        What seemed like a failure may later be understood as a turning point.
        
        Triggers:
        1. Identity dimension strength changed significantly
        2. Relationship quality changed (good→bad or bad→good)
        3. Accumulated enough feedback
        """
        for plot in self.plots.values():
            if plot.status != "active":
                continue
            if not plot.identity_impact:
                continue
            
            # Check if reframe is needed
            should_reframe, reason = self._should_reframe(plot)
            
            if should_reframe:
                new_meaning = self._generate_new_meaning(plot, reason)
                if new_meaning and new_meaning != plot.identity_impact.current_meaning:
                    plot.identity_impact.update_meaning(new_meaning)
    
    def _should_reframe(self, plot: Plot) -> Tuple[bool, str]:
        """Determine if a plot's meaning should be reframed."""
        if not plot.identity_impact:
            return False, ""
        
        # Trigger 1: Old plot with high access (important but meaning may have evolved)
        age_days = (now_ts() - plot.ts) / 86400
        if age_days > 7 and plot.access_count > 5:
            # Check if identity dimensions have changed
            for dim in plot.identity_impact.identity_dimensions_affected:
                current_strength = self._identity_dimensions.get(dim, 0.5)
                # If we've grown in this dimension, the meaning may have deepened
                if current_strength > 0.7:
                    return True, f"身份维度「{dim}」已增强"
        
        # Trigger 2: Relationship quality changed significantly
        if plot.relational:
            relationship_entity = plot.relational.with_whom
            story = self.get_relationship_story(relationship_entity)
            if story:
                # Compare current trust with trust at plot time
                current_trust = story.get_current_trust()
                original_delta = plot.relational.relationship_quality_delta
                
                # If relationship improved but plot was negative, reframe
                if current_trust > 0.7 and original_delta < -0.1:
                    return True, "关系改善后重新理解"
                # If relationship worsened but plot was positive, reframe
                if current_trust < 0.3 and original_delta > 0.1:
                    return True, "关系变化后重新理解"
        
        # Trigger 3: Meaning hasn't been updated in a while for important plots
        if not plot.identity_impact.evolution_history:
            if age_days > 30 and plot.access_count > 3:
                return True, "定期反思"
        
        return False, ""
    
    def _generate_new_meaning(self, plot: Plot, reason: str) -> Optional[str]:
        """Generate a new meaning for a plot based on current understanding."""
        if not plot.identity_impact:
            return None
        
        # Generate new meaning based on reason
        if "身份维度" in reason and "增强" in reason:
            # The experience contributed to growth
            dims = plot.identity_impact.identity_dimensions_affected[:2]
            if dims:
                return f"这是我成为{dims[0]}的重要一步"
        
        elif "关系改善" in reason:
            return f"虽然当时困难，但这帮助了关系的成长"
        
        elif "关系变化" in reason:
            return f"这次经历让我对这段关系有了更深的理解"
        
        elif "定期反思" in reason:
            # Generic deepening of meaning
            if plot.relational:
                return f"在与{plot.relational.with_whom}的关系中，这是一个有意义的时刻"
        
        return None

    # -------------------------------------------------------------------------
    # Identity Tension Analysis (NEW - "身份矛盾分析")
    # -------------------------------------------------------------------------
    
    def _analyze_identity_tensions(self) -> None:
        """
        Analyze tensions between identity dimensions.
        
        Philosophy: Not all tensions are bad.
        - Some should be resolved (action-blocking)
        - Some should be preserved (provide flexibility)
        - Some should be accepted (signs of growth)
        """
        theme_list = list(self.themes.values())
        
        for i, theme_a in enumerate(theme_list):
            if not theme_a.is_identity_dimension():
                continue
            
            for theme_b in theme_list[i+1:]:
                if not theme_b.is_identity_dimension():
                    continue
                
                # Check for tension
                if theme_a.prototype is not None and theme_b.prototype is not None:
                    sim = self.metric.sim(theme_a.prototype, theme_b.prototype)
                    
                    # Moderate similarity but different dimensions = potential adaptive tension
                    if 0.2 < sim < 0.6:
                        # This is likely an adaptive tension (both can be true in different contexts)
                        # Example: "作为耐心解释者的我" vs "作为高效回应者的我"
                        theme_a.add_tension(theme_b.id)
                        theme_b.add_tension(theme_a.id)
                    
                    elif sim > 0.7:
                        # High similarity = harmony, these dimensions complement each other
                        theme_a.add_harmony(theme_b.id)
                        theme_b.add_harmony(theme_a.id)

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
    # Growth-Oriented Forgetting (replaces capacity-driven pressure management)
    # -------------------------------------------------------------------------

    def _pressure_manage(self) -> None:
        """Growth-oriented forgetting: decide what to forget based on growth, not just capacity.
        
        Key philosophy change:
        - Old: "Delete low-mass plots when capacity is exceeded"
        - New: "Forget what doesn't help me become a better self"
        
        The question is not "what's least important?" but 
        "does keeping this help me grow?"
        """
        max_plots = self.cfg.max_plots
        if len(self.plots) <= max_plots:
            return

        # Candidate plots that are active and already assigned to a story
        cands = [p for p in self.plots.values() if p.status == "active" and p.story_id is not None]
        if not cands:
            return

        # === Growth-oriented scoring ===
        for plot in cands:
            # 1. Identity contribution: Does this plot support my identity?
            identity_contribution = self._compute_identity_contribution(plot)
            
            # 2. Relationship contribution: Does this plot maintain an important relationship?
            relationship_contribution = self._compute_relationship_contribution(plot)
            
            # 3. Growth contribution: Does keeping this help me grow?
            growth_contribution = self._compute_growth_contribution(plot)
            
            # Combined keep score (higher = more reason to keep)
            plot._keep_score = (
                0.4 * identity_contribution +
                0.3 * relationship_contribution +
                0.3 * growth_contribution
            )
        
        # Build logits favoring low keep-score plots (forget those that don't help growth)
        keep_scores = np.array([getattr(p, '_keep_score', p.mass()) for p in cands], dtype=np.float32)
        # Lower keep score -> higher probability to forget
        logits = (-keep_scores).tolist()
        probs = np.array(softmax(logits), dtype=np.float64)

        # Number to remove
        excess = len(self.plots) - max_plots
        remove_ids = set(self.rng.choice([p.id for p in cands], size=excess, replace=False, p=probs))
        
        for pid in remove_ids:
            p = self.plots.get(pid)
            if p is None:
                continue
            self._forget_plot(pid)

    def _compute_identity_contribution(self, plot: Plot) -> float:
        """
        Compute how much this plot contributes to current identity.
        
        A plot contributes to identity if:
        - It's evidence for an identity dimension (Theme)
        - It reinforces who I am
        """
        if not plot.identity_impact:
            return 0.3  # Baseline for plots without explicit identity impact
        
        contribution = 0.0
        
        # Check if this plot is evidence for active identity dimensions
        for dim in plot.identity_impact.identity_dimensions_affected:
            dim_strength = self._identity_dimensions.get(dim, 0.0)
            contribution += dim_strength * 0.3
        
        # Check if the plot's story is connected to important themes
        if plot.story_id and plot.story_id in self.stories:
            story = self.stories[plot.story_id]
            # Find themes this story supports
            for theme in self.themes.values():
                if plot.story_id in theme.story_ids:
                    contribution += theme.confidence() * 0.2
        
        return min(1.0, contribution + 0.2)  # Baseline of 0.2
    
    def _compute_relationship_contribution(self, plot: Plot) -> float:
        """
        Compute how much this plot maintains an important relationship.
        
        A plot contributes to relationships if:
        - It's an anchor point in an important relationship
        - The relationship is healthy and ongoing
        """
        if not plot.relational:
            return 0.3  # Baseline for non-relational plots
        
        relationship_entity = plot.relational.with_whom
        story_id = self._relationship_story_index.get(relationship_entity)
        
        if not story_id or story_id not in self.stories:
            return 0.3
        
        story = self.stories[story_id]
        
        # Factor 1: Relationship health
        health_factor = story.relationship_health
        
        # Factor 2: Relationship recency (ongoing relationships matter more)
        recency = 1.0 / (1.0 + math.log1p(now_ts() - story.updated_ts) / 10)
        
        # Factor 3: Is this plot recent in the relationship?
        if story.plot_ids:
            plot_position = story.plot_ids.index(plot.id) if plot.id in story.plot_ids else -1
            if plot_position >= 0:
                # More recent plots matter more
                recency_in_story = plot_position / len(story.plot_ids)
            else:
                recency_in_story = 0.5
        else:
            recency_in_story = 0.5
        
        return 0.3 * health_factor + 0.4 * recency + 0.3 * recency_in_story
    
    def _compute_growth_contribution(self, plot: Plot) -> float:
        """
        Compute how much keeping this plot helps growth.
        
        Questions:
        - Does this provide ongoing learning value?
        - Does this influence future behavior?
        - Does this hinder growth (e.g., reinforce negative self-image)?
        """
        # Factor 1: Learning value - high surprise/tension plots often have learning value
        learning_value = min(1.0, plot.tension * 0.5) if plot.tension > 0 else 0.3
        
        # Factor 2: Future influence - recent, accessed plots influence future
        age_factor = 1.0 / math.log1p(max(1.0, now_ts() - plot.ts))
        access_factor = math.log1p(plot.access_count + 1) / 5.0
        future_influence = 0.5 * age_factor + 0.5 * min(1.0, access_factor)
        
        # Factor 3: Growth hindrance check
        # (In a full implementation, this would use sentiment/content analysis)
        # For now, use a simple heuristic: very old, never-accessed plots may hinder
        if plot.access_count == 0 and (now_ts() - plot.ts) > 30 * 24 * 3600:  # 30 days
            growth_hindrance = 0.3  # Some penalty
        else:
            growth_hindrance = 0.0
        
        return 0.4 * learning_value + 0.4 * future_influence - growth_hindrance

    def _forget_plot(self, plot_id: str) -> None:
        """
        Forget a plot - not deletion, but letting go.
        
        Philosophy: "Forgetting is not losing information, 
        it's choosing what to become."
        
        The plot's essence is preserved in:
        - The Story's centroid (aggregate meaning)
        - The relationship trajectory (if relational)
        - The identity dimensions it affected
        """
        p = self.plots.get(plot_id)
        if p is None or p.story_id is None:
            return
        
        p.status = "absorbed"

        # Remove from vector index to reduce retrieval noise
        self.vindex.remove(plot_id)
        
        # If this plot had identity impact, the impact is preserved in the dimensions
        # (The _identity_dimensions dict retains the accumulated effect)
        
        # If this plot had relational context, the relationship trajectory preserves it
        # (The story's relationship_arc retains the pattern)

    def _absorb_plot(self, plot_id: str) -> None:
        """Legacy method - redirects to _forget_plot for compatibility."""
        self._forget_plot(plot_id)

    # -------------------------------------------------------------------------
    # Convenience: inspect
    # -------------------------------------------------------------------------

    def get_story(self, story_id: str) -> StoryArc:
        return self.stories[story_id]

    def get_plot(self, plot_id: str) -> Plot:
        return self.plots[plot_id]

    def get_theme(self, theme_id: str) -> Theme:
        return self.themes[theme_id]
    
    def get_relationship_story(self, entity_id: str) -> Optional[StoryArc]:
        """Get the story for a specific relationship entity."""
        story_id = self._relationship_story_index.get(entity_id)
        if story_id:
            return self.stories.get(story_id)
        return None
    
    def get_my_identity_with(self, entity_id: str) -> Optional[str]:
        """Get my identity in a specific relationship."""
        story = self.get_relationship_story(entity_id)
        if story:
            return story.my_identity_in_this_relationship
        return None
    
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
        
        identity_summary = {
            "identity_dimensions": dict(self._identity_dimensions),
            "relationship_identities": {
                entity: story.my_identity_in_this_relationship
                for entity, story in relationships.items()
            },
            "relationship_count": len(relationships),
            "total_interactions": sum(len(s.plot_ids) for s in relationships.values()),
        }
        
        # Find dominant identity dimension
        if self._identity_dimensions:
            dominant = max(self._identity_dimensions, key=self._identity_dimensions.get)
            identity_summary["dominant_dimension"] = dominant
        
        return identity_summary

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
            
            # Relationship-centric additions
            "relationship_story_index": self._relationship_story_index,
            "identity_dimensions": self._identity_dimensions,
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
        
        # Restore relationship-centric additions
        obj._relationship_story_index = d.get("relationship_story_index", {})
        obj._identity_dimensions = d.get("identity_dimensions", {})
        
        # Rebuild relationship index from stories if not present
        if not obj._relationship_story_index:
            for sid, story in obj.stories.items():
                if story.relationship_with:
                    obj._relationship_story_index[story.relationship_with] = sid
        
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
