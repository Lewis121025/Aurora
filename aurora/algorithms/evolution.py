"""
AURORA Evolution Module
=======================

Evolution, reflection, and meaning reframe functionality.

Key responsibilities:
- Relationship reflection (assess relationship health, extract lessons)
- Meaning reframe (update interpretations of past experiences)
- Story status updates (probabilistic based on activity)
- Theme/identity dimension emergence
- Identity tension analysis
- Story boundary detection (climax, resolution, abandonment)
- Graph structure cleanup (weak edges, similar nodes, stale content)

Philosophy: "持续成为" (continuous becoming) - identity emerges and evolves through experiences.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np

from aurora.algorithms.constants import (
    ARCHIVE_MIN_ACCESS_COUNT,
    ARCHIVE_STALE_DAYS_THRESHOLD,
    CLIMAX_DECLINE_RATIO,
    CLIMAX_TENSION_WINDOW,
    EPSILON_PRIOR,
    HARMONY_SIMILARITY_MIN,
    NODE_MERGE_SIMILARITY_THRESHOLD,
    PERIODIC_REFLECTION_ACCESS_COUNT,
    PERIODIC_REFLECTION_AGE_DAYS,
    REFRAME_ACCESS_COUNT_THRESHOLD,
    REFRAME_AGE_DAYS_THRESHOLD,
    RESOLUTION_MIN_ARC_LENGTH,
    RESOLUTION_TENSION_DROP_RATIO,
    STORY_ABANDONMENT_THRESHOLD_DAYS,
    TENSION_SIMILARITY_MAX,
    TENSION_SIMILARITY_MIN,
    WEAK_EDGE_MIN_SUCCESSES,
    WEAK_EDGE_MIN_WEIGHT,
)
from aurora.algorithms.models.theme import Theme
from aurora.algorithms.models.trace import EvolutionPatch, EvolutionSnapshot
from aurora.utils.id_utils import det_id
from aurora.utils.math_utils import sigmoid, softmax
from aurora.utils.time_utils import now_ts

if TYPE_CHECKING:
    from aurora.algorithms.models.plot import Plot
    from aurora.algorithms.models.story import StoryArc

logger = logging.getLogger(__name__)


class EvolutionMixin:
    """Mixin providing evolution, reflection, and meaning reframe functionality."""

    # -------------------------------------------------------------------------
    # Relationship Reflection ("关系反思")
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
            if role_consistency > HARMONY_SIMILARITY_MIN and story.relationship_arc:
                dominant_role = story.relationship_arc[-1].my_role
                if dominant_role != story.my_identity_in_this_relationship:
                    story.update_identity_in_relationship(dominant_role)
    
    def _extract_relationship_lesson(self, story: "StoryArc") -> Optional[str]:
        """
        Extract a lesson from relationship history.
        
        Args:
            story: The relationship story to extract lessons from
            
        Returns:
            A lesson string or None if no clear lesson
        """
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
    # Meaning Reframe ("意义重构")
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
            should_reframe, reason = self._check_should_reframe(plot)
            
            if should_reframe:
                new_meaning = self._generate_new_meaning(plot, reason)
                if new_meaning and new_meaning != plot.identity_impact.current_meaning:
                    plot.identity_impact.update_meaning(new_meaning)
    
    def _check_should_reframe(self, plot: "Plot") -> Tuple[bool, str]:
        """
        Determine if a plot's meaning should be reframed.
        
        Args:
            plot: The plot to check
            
        Returns:
            Tuple of (should_reframe: bool, reason: str)
        """
        if not plot.identity_impact:
            return False, ""
        
        # Trigger 1: Old plot with high access (important but meaning may have evolved)
        age_days = (now_ts() - plot.ts) / 86400
        if age_days > REFRAME_AGE_DAYS_THRESHOLD and plot.access_count > REFRAME_ACCESS_COUNT_THRESHOLD:
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
            if age_days > PERIODIC_REFLECTION_AGE_DAYS and plot.access_count > PERIODIC_REFLECTION_ACCESS_COUNT:
                return True, "定期反思"
        
        return False, ""
    
    def _generate_new_meaning(self, plot: "Plot", reason: str) -> Optional[str]:
        """
        Generate a new meaning for a plot based on current understanding.
        
        Args:
            plot: The plot to generate new meaning for
            reason: The reason for reframing
            
        Returns:
            New meaning string or None
        """
        if not plot.identity_impact:
            return None
        
        # Generate new meaning based on reason
        if "身份维度" in reason and "增强" in reason:
            # The experience contributed to growth
            dims = plot.identity_impact.identity_dimensions_affected[:2]
            if dims:
                return f"这是我成为{dims[0]}的重要一步"
        
        elif "关系改善" in reason:
            return "虽然当时困难，但这帮助了关系的成长"
        
        elif "关系变化" in reason:
            return "这次经历让我对这段关系有了更深的理解"
        
        elif "定期反思" in reason:
            # Generic deepening of meaning
            if plot.relational:
                return f"在与{plot.relational.with_whom}的关系中，这是一个有意义的时刻"
        
        return None

    # -------------------------------------------------------------------------
    # Identity Tension Analysis ("身份矛盾分析")
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
                    if TENSION_SIMILARITY_MIN < sim < TENSION_SIMILARITY_MAX:
                        # This is likely an adaptive tension (both can be true in different contexts)
                        # Example: "作为耐心解释者的我" vs "作为高效回应者的我"
                        theme_a.add_tension(theme_b.id)
                        theme_b.add_tension(theme_a.id)
                    
                    elif sim > HARMONY_SIMILARITY_MIN:
                        # High similarity = harmony, these dimensions complement each other
                        theme_a.add_harmony(theme_b.id)
                        theme_b.add_harmony(theme_a.id)

    # -------------------------------------------------------------------------
    # Story Boundary Detection ("故事边界检测")
    # -------------------------------------------------------------------------

    def _detect_story_boundaries(self) -> None:
        """
        Detect story boundaries: climax, resolution, and abandonment.
        
        Philosophy: Stories have natural lifecycles. Detecting where a story
        is in its lifecycle helps with memory organization and retrieval.
        
        Three key boundary types:
        - Climax: The story reaches peak tension (transformative moment)
        - Resolution: Core conflicts are resolved (natural ending)
        - Abandonment: Story goes dormant (may be revisited later)
        """
        boundary_stats = {"climax": 0, "resolution": 0, "abandonment": 0}
        
        for story in self.stories.values():
            if story.status != "developing":
                continue
            
            # Check for climax (peak tension reached)
            if self._detect_climax(story):
                story.status = "resolved"
                boundary_stats["climax"] += 1
                logger.debug(f"Story {story.id} reached climax, marking as resolved")
                continue
            
            # Check for resolution (conflict resolved)
            if self._detect_resolution(story):
                story.status = "resolved"
                boundary_stats["resolution"] += 1
                logger.debug(f"Story {story.id} reached resolution")
                continue
            
            # Check for abandonment (long inactivity)
            if self._detect_abandonment(story):
                story.status = "abandoned"
                boundary_stats["abandonment"] += 1
                logger.debug(f"Story {story.id} detected as abandoned")
        
        if any(boundary_stats.values()):
            logger.info(f"Story boundary detection: {boundary_stats}")

    def _detect_climax(self, story: "StoryArc") -> bool:
        """
        Detect if a story has reached its climax (peak tension followed by decline).
        
        Philosophy: A climax is a transformative moment where tension peaks
        and then starts to release. This is probabilistic, not deterministic.
        
        Args:
            story: The story to check
            
        Returns:
            True if climax detected with probabilistic certainty
        """
        curve = story.tension_curve
        
        # Need sufficient history to detect climax
        if len(curve) < CLIMAX_TENSION_WINDOW:
            return False
        
        # Find peak in the tension curve
        window = curve[-CLIMAX_TENSION_WINDOW:]
        peak_idx = int(np.argmax(window))
        peak_val = window[peak_idx]
        
        # Peak should be at least 30% higher than current
        current_val = window[-1]
        if peak_val <= 0:
            return False
        
        decline_ratio = (peak_val - current_val) / peak_val
        
        # Use sigmoid to convert to probability instead of hard threshold
        # Higher decline = higher probability of climax
        p_climax = sigmoid((decline_ratio - CLIMAX_DECLINE_RATIO) * 10)
        
        # Probabilistic decision
        return self.rng.random() < p_climax

    def _detect_resolution(self, story: "StoryArc") -> bool:
        """
        Detect if a story has reached resolution (core conflict resolved).
        
        Philosophy: Resolution occurs when tension drops significantly from
        its peak, indicating the core conflict has been addressed.
        
        Args:
            story: The story to check
            
        Returns:
            True if resolution detected with probabilistic certainty
        """
        curve = story.tension_curve
        
        # Need sufficient arc length for meaningful resolution
        if len(curve) < RESOLUTION_MIN_ARC_LENGTH:
            return False
        
        # Check if tension has dropped significantly from peak
        peak_tension = max(curve)
        current_tension = curve[-1]
        
        if peak_tension <= 0:
            return False
        
        drop_ratio = (peak_tension - current_tension) / peak_tension
        
        # Additional signal: check relationship health for relationship stories
        health_factor = 1.0
        if story.is_relationship_story():
            # High relationship health increases resolution probability
            health_factor = 1.0 + (story.relationship_health - 0.5)
        
        # Use sigmoid to convert to probability
        p_resolution = sigmoid((drop_ratio - RESOLUTION_TENSION_DROP_RATIO) * 8) * health_factor
        p_resolution = min(1.0, p_resolution)  # Clamp to valid probability
        
        return self.rng.random() < p_resolution

    def _detect_abandonment(
        self, story: "StoryArc", threshold_days: float = STORY_ABANDONMENT_THRESHOLD_DAYS
    ) -> bool:
        """
        Detect if a story has been abandoned (long period without activity).
        
        Philosophy: Stories can go dormant without reaching a clear resolution.
        This isn't necessarily negative - they may be revisited later.
        Abandonment is probabilistic based on inactivity duration.
        
        Args:
            story: The story to check
            threshold_days: Days of inactivity before considering abandonment
            
        Returns:
            True if abandonment detected with probabilistic certainty
        """
        ts = now_ts()
        idle_seconds = ts - story.updated_ts
        idle_days = idle_seconds / 86400.0
        
        # Use story's own temporal rhythm as context
        # Stories that typically have long gaps are less likely to be "abandoned"
        gap_mean_days = story.gap_mean_safe() / 86400.0
        
        # Normalize idle time by the story's typical gap
        # If idle_days >> gap_mean_days, more likely abandoned
        if gap_mean_days > 0:
            normalized_idle = idle_days / gap_mean_days
        else:
            normalized_idle = idle_days / 7.0  # Default 1 week rhythm
        
        # Calculate abandonment probability using sigmoid
        # Center at threshold_days, with story rhythm as modulation
        p_abandon = sigmoid((idle_days - threshold_days) / 10.0)
        
        # Reduce probability if story has active relationship context
        if story.is_relationship_story() and story.relationship_health > 0.6:
            p_abandon *= 0.5  # Healthy relationships are less likely to be abandoned
        
        return self.rng.random() < p_abandon

    # -------------------------------------------------------------------------
    # Graph Structure Cleanup ("图结构清理")
    # -------------------------------------------------------------------------

    def _cleanup_graph_structure(self) -> None:
        """
        Clean up graph structure for memory efficiency and coherence.
        
        Philosophy: Memory systems need maintenance to stay healthy.
        Like human forgetting, this is adaptive, not pathological.
        
        Cleanup operations:
        1. Remove weak edges (low confidence connections)
        2. Merge highly similar nodes (reduce redundancy)
        3. Archive stale content (long-unused memories)
        """
        logger.info("Starting graph structure cleanup")
        
        # 1. Remove weak edges
        edges_removed = self._remove_weak_edges()
        
        # 2. Merge similar nodes (plots only, to preserve structure)
        nodes_merged = self._merge_similar_nodes()
        
        # 3. Archive stale content
        archived = self._archive_stale_content()
        
        logger.info(
            f"Graph cleanup complete: edges_removed={edges_removed}, "
            f"nodes_merged={nodes_merged}, content_archived={archived}"
        )

    def _remove_weak_edges(self, min_weight: float = WEAK_EDGE_MIN_WEIGHT) -> int:
        """
        Remove edges with low belief weight (low confidence connections).
        
        Philosophy: Edges that haven't been reinforced through use
        are candidates for removal. Use probabilistic decision to
        avoid harsh cutoffs.
        
        Args:
            min_weight: Minimum edge weight to keep (soft threshold)
            
        Returns:
            Number of edges removed
        """
        edges_to_remove: List[Tuple[str, str]] = []
        
        for src, dst, data in self.graph.g.edges(data=True):
            belief = data.get("belief")
            if belief is None:
                continue
            
            # Get edge strength (probability of positive outcome)
            edge_weight = belief.mean()
            
            # Calculate removal probability using sigmoid
            # Edges well below threshold have high removal probability
            p_remove = sigmoid((min_weight - edge_weight) * 20)
            
            # Additional factor: edges with very few trials are kept
            # (benefit of the doubt for new edges)
            # EdgeBelief uses Beta(a, b) where a-1 = successes, b-1 = failures
            # use_count tracks total uses
            if belief.use_count < WEAK_EDGE_MIN_SUCCESSES:
                p_remove *= 0.1  # Strong reduction in removal probability
            
            if self.rng.random() < p_remove:
                edges_to_remove.append((src, dst))
        
        # Remove edges
        for src, dst in edges_to_remove:
            self.graph.g.remove_edge(src, dst)
            logger.debug(f"Removed weak edge: {src} -> {dst}")
        
        return len(edges_to_remove)

    def _merge_similar_nodes(
        self, similarity_threshold: float = NODE_MERGE_SIMILARITY_THRESHOLD
    ) -> int:
        """
        Merge highly similar plot nodes to reduce redundancy.
        
        Philosophy: Very similar memories can be consolidated without
        losing essential information. This is analogous to human
        memory consolidation during sleep.
        
        Args:
            similarity_threshold: Similarity above which nodes may be merged
            
        Returns:
            Number of merge operations performed
        """
        merge_count = 0
        merged_ids: Set[str] = set()
        
        # Only merge plots (stories and themes have semantic structure)
        plot_ids = list(self.plots.keys())
        
        for i, pid_a in enumerate(plot_ids):
            if pid_a in merged_ids:
                continue
            
            plot_a = self.plots.get(pid_a)
            if plot_a is None or plot_a.status != "active":
                continue
            
            for pid_b in plot_ids[i + 1:]:
                if pid_b in merged_ids:
                    continue
                
                plot_b = self.plots.get(pid_b)
                if plot_b is None or plot_b.status != "active":
                    continue
                
                # Calculate similarity
                sim = self.metric.sim(plot_a.embedding, plot_b.embedding)
                
                # Probabilistic merge decision
                p_merge = sigmoid((sim - similarity_threshold) * 50)
                
                if self.rng.random() < p_merge:
                    # Merge B into A (keep A as the survivor)
                    self._merge_plots(plot_a, plot_b)
                    merged_ids.add(pid_b)
                    merge_count += 1
                    logger.debug(f"Merged plot {pid_b} into {pid_a} (sim={sim:.3f})")
        
        return merge_count

    def _merge_plots(self, survivor: "Plot", merged: "Plot") -> None:
        """
        Merge one plot into another.
        
        Args:
            survivor: The plot that will remain
            merged: The plot to merge into survivor
        """
        # Update survivor with merged data
        survivor.access_count += merged.access_count
        
        # Keep the more recent last_access_ts
        if merged.last_access_ts and survivor.last_access_ts:
            survivor.last_access_ts = max(survivor.last_access_ts, merged.last_access_ts)
        elif merged.last_access_ts:
            survivor.last_access_ts = merged.last_access_ts
        
        # Mark merged plot as absorbed (semantically: absorbed into another plot)
        merged.status = "absorbed"
        
        # Update story references if needed
        if merged.story_id and merged.story_id in self.stories:
            story = self.stories[merged.story_id]
            if merged.id in story.plot_ids:
                story.plot_ids.remove(merged.id)
                if survivor.id not in story.plot_ids:
                    story.plot_ids.append(survivor.id)
        
        # Remove merged plot from vector index
        self.vindex.remove(merged.id)
        
        # Update graph edges: redirect edges from merged to survivor
        if self.graph.g.has_node(merged.id):
            # Get all edges involving merged node
            in_edges = list(self.graph.g.in_edges(merged.id))
            out_edges = list(self.graph.g.out_edges(merged.id))
            
            # Redirect edges to survivor
            for src, _ in in_edges:
                if src != survivor.id:
                    edge_data = self.graph.g.edges[src, merged.id]
                    self.graph.g.add_edge(src, survivor.id, **edge_data)
            
            for _, dst in out_edges:
                if dst != survivor.id:
                    edge_data = self.graph.g.edges[merged.id, dst]
                    self.graph.g.add_edge(survivor.id, dst, **edge_data)
            
            # Remove merged node
            self.graph.g.remove_node(merged.id)
        
        # Remove from plots dict
        del self.plots[merged.id]

    def _archive_stale_content(
        self, days_threshold: float = ARCHIVE_STALE_DAYS_THRESHOLD
    ) -> int:
        """
        Archive content that hasn't been accessed for a long time.
        
        Philosophy: Long-unused memories are archived (not deleted) to
        maintain system performance. They can be retrieved if needed
        but don't participate in active retrieval.
        
        Args:
            days_threshold: Days without access before archiving (soft threshold)
            
        Returns:
            Number of items archived
        """
        ts = now_ts()
        archived_count = 0
        
        for plot in self.plots.values():
            if plot.status != "active":
                continue
            
            # Calculate staleness
            if plot.last_access_ts:
                idle_days = (ts - plot.last_access_ts) / 86400.0
            else:
                idle_days = (ts - plot.ts) / 86400.0
            
            # Access count affects archival probability
            # Frequently accessed content is less likely to be archived
            access_factor = 1.0 / (1.0 + plot.access_count)
            
            # Calculate archival probability
            p_archive = sigmoid((idle_days - days_threshold) / 30.0) * access_factor
            
            # Very low access count increases archival probability
            if plot.access_count <= ARCHIVE_MIN_ACCESS_COUNT:
                p_archive = min(1.0, p_archive * 1.5)
            
            if self.rng.random() < p_archive:
                plot.status = "archived"
                archived_count += 1
                
                # Remove from vector index to speed up searches
                self.vindex.remove(plot.id)
                
                logger.debug(
                    f"Archived plot {plot.id} (idle_days={idle_days:.1f}, "
                    f"access_count={plot.access_count})"
                )
        
        return archived_count

    # -------------------------------------------------------------------------
    # Async Evolution: Copy-on-Write for non-blocking processing
    # -------------------------------------------------------------------------

    def create_evolution_snapshot(self) -> EvolutionSnapshot:
        """
        Create a read-only snapshot for evolution processing.
        
        This captures the current state without holding locks, allowing
        evolution to proceed in a background thread while new ingests continue.
        
        Returns:
            EvolutionSnapshot containing current state
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

    def compute_evolution_patch(self, snapshot: EvolutionSnapshot) -> EvolutionPatch:
        """Compute evolution changes from snapshot (pure function, no side effects)."""
        rng = np.random.default_rng()
        rng.bit_generator.state = snapshot.rng_state
        
        status_changes = self._compute_story_status_changes(snapshot, rng)
        theme_assignments, new_themes = self._compute_theme_assignments(snapshot, status_changes, rng)
        
        return EvolutionPatch(
            status_changes=status_changes,
            theme_assignments=theme_assignments,
            new_themes=new_themes,
        )

    def _compute_story_status_changes(
        self, snapshot: EvolutionSnapshot, rng: np.random.Generator
    ) -> Dict[str, str]:
        """Compute which stories should change status."""
        status_changes: Dict[str, str] = {}
        ts = now_ts()
        
        for sid in snapshot.story_ids:
            if snapshot.story_statuses[sid] != "developing":
                continue
            
            # Compute activity probability
            updated = snapshot.story_updated_ts[sid]
            tau = snapshot.story_gap_means[sid]
            idle = max(0.0, ts - updated)
            p_active = math.exp(-idle / max(tau, EPSILON_PRIOR))
            
            if rng.random() < p_active:
                continue
            
            # Determine resolve vs abandon
            curve = snapshot.story_tension_curves[sid]
            p_resolve = sigmoid(-(curve[-1] - curve[0])) if len(curve) >= 3 else 0.5
            status_changes[sid] = "resolved" if rng.random() < p_resolve else "abandoned"
        
        return status_changes

    def _compute_theme_assignments(
        self,
        snapshot: EvolutionSnapshot,
        status_changes: Dict[str, str],
        rng: np.random.Generator,
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, np.ndarray]]]:
        """Compute theme assignments for newly resolved stories."""
        theme_assignments: List[Tuple[str, str]] = []
        new_themes: List[Tuple[str, np.ndarray]] = []
        current_theme_counts = dict(snapshot.theme_story_counts)
        
        for sid, new_status in status_changes.items():
            if new_status != "resolved":
                continue
            
            centroid = snapshot.story_centroids[sid]
            if centroid is None:
                continue
            
            choice = self._sample_theme_assignment(
                centroid, snapshot, current_theme_counts, rng
            )
            
            if choice == "__new__":
                new_theme_id = det_id("theme", sid)
                new_themes.append((new_theme_id, centroid.copy()))
                theme_assignments.append((sid, new_theme_id))
                current_theme_counts[new_theme_id] = 1
            else:
                theme_assignments.append((sid, choice))
                current_theme_counts[choice] = current_theme_counts.get(choice, 0) + 1
        
        return theme_assignments, new_themes

    def _sample_theme_assignment(
        self,
        centroid: np.ndarray,
        snapshot: EvolutionSnapshot,
        current_counts: Dict[str, int],
        rng: np.random.Generator,
    ) -> str:
        """Sample a theme assignment using CRP."""
        logps: Dict[str, float] = {}
        
        for tid in snapshot.theme_ids:
            prior = math.log(current_counts.get(tid, 0) + EPSILON_PRIOR)
            prototype = snapshot.theme_prototypes.get(tid)
            if prototype is not None:
                d2 = float(np.dot(centroid - prototype, centroid - prototype))
                logps[tid] = prior - 0.5 * d2
            else:
                logps[tid] = prior
        
        logps["__new__"] = math.log(snapshot.crp_theme_alpha)
        
        keys = list(logps.keys())
        probs = softmax([logps[k] for k in keys])
        return rng.choice(keys, p=np.array(probs, dtype=np.float64))

    def apply_evolution_patch(self, patch: EvolutionPatch) -> None:
        """
        Apply computed evolution changes atomically.
        
        Should be called with appropriate locking if needed.
        
        Args:
            patch: EvolutionPatch to apply
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
                theme.prototype = self._update_centroid_online(
                    theme.prototype, story.centroid, len(theme.story_ids)
                )
            
            # Weave edges
            self._create_bidirectional_edge(sid, tid, "thematizes", "exemplified_by")
        
        # 4) Pressure management
        self._pressure_manage()
