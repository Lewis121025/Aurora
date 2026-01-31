"""
AURORA Relationship Module
==========================

Relationship identification, identity assessment, and relational context extraction.

Key responsibilities:
- Identify relationship entities from interactions
- Assess identity relevance of interactions
- Extract relational context ("who I am in this relationship")
- Extract identity impact ("how this affects who I am")

Philosophy: Memory should be organized around relationships, not just semantic similarity.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Tuple

from aurora.algorithms.constants import (
    CHALLENGE_WEIGHT,
    IDENTITY_DIMENSION_GROWTH_RATE,
    IDENTITY_RELEVANCE_THRESHOLD,
    INTERACTION_COUNT_LOG_NORMALIZER,
    MAX_IDENTITY_DIMENSIONS,
    MODERATE_SIMILARITY_MAX,
    MODERATE_SIMILARITY_MIN,
    NOVELTY_WEIGHT,
    QUALITY_DELTA_COEFFICIENT,
    MAX_QUALITY_DELTA,
    REINFORCEMENT_WEIGHT,
    ROLE_CONSISTENCY_THRESHOLD,
)
from aurora.algorithms.models.plot import IdentityImpact, Plot, RelationalContext
from aurora.algorithms.models.story import StoryArc
from aurora.utils.id_utils import det_id
from aurora.utils.time_utils import now_ts

if TYPE_CHECKING:
    import numpy as np


class RelationshipMixin:
    """Mixin providing relationship identification and identity assessment functionality."""

    # -------------------------------------------------------------------------
    # Relationship Identification
    # -------------------------------------------------------------------------

    def _identify_relationship_entity(self, actors: Tuple[str, ...], text: str) -> str:
        """
        Identify the primary relationship entity from this interaction.
        
        The key insight: memory should be organized around relationships,
        not just semantic similarity. This method identifies who the "other"
        is in this interaction.
        
        Args:
            actors: Tuple of actors involved in the interaction
            text: Interaction text (for context, not currently used)
            
        Returns:
            The identified relationship entity name
        """
        # Filter out the agent/assistant itself
        others = [a for a in actors if a.lower() not in ("agent", "assistant", "ai", "system")]
        
        if others:
            # Return the first non-self actor as the relationship entity
            return others[0]
        
        # If no clear other, use "user" as default
        return "user"

    # -------------------------------------------------------------------------
    # Identity Relevance Assessment
    # -------------------------------------------------------------------------

    def _assess_identity_relevance(
        self, text: str, relationship_entity: str, emb: "np.ndarray"
    ) -> float:
        """
        Assess how relevant this interaction is to "who I am".
        
        This replaces pure information-theoretic VOI with identity-relevance.
        The question is not "is this surprising?" but "does this affect my identity?"
        
        Optimized: Single traversal computes reinforcement, challenge, and novelty.
        
        Args:
            text: Interaction text
            relationship_entity: The identified relationship entity
            emb: Embedding of the interaction
            
        Returns:
            Identity relevance score between 0 and 1
        """
        # Compute all identity signals in one pass
        reinforcement, challenge, novelty = self._compute_identity_signals(text, emb)
        
        relevance = (
            reinforcement * REINFORCEMENT_WEIGHT +
            challenge * CHALLENGE_WEIGHT +  # Challenges are more important to remember
            novelty * NOVELTY_WEIGHT
        )
        
        # Relationship factor: interactions in important relationships matter more
        relationship_importance = self._get_relationship_importance(relationship_entity)
        relevance *= (0.5 + 0.5 * relationship_importance)
        
        return min(1.0, relevance)

    def _compute_identity_signals(
        self, text: str, emb: "np.ndarray"
    ) -> Tuple[float, float, float]:
        """
        Compute identity signals in a single traversal of themes.
        
        This optimized method computes reinforcement, challenge, and novelty
        in a single pass over all themes, avoiding redundant computations.
        
        Args:
            text: Interaction text
            emb: Embedding of the interaction
            
        Returns:
            Tuple of (reinforcement, challenge, novelty) scores
        """
        # Early returns for edge cases
        if not self.themes:
            return 0.0, 0.0, 1.0  # No themes = everything is novel
        
        # Single pass over all themes
        max_sim = 0.0
        min_sim = 1.0
        max_moderate_sim = 0.0
        
        for theme in self.themes.values():
            if theme.prototype is None:
                continue
            sim = self.metric.sim(emb, theme.prototype)
            
            # For reinforcement: max similarity
            max_sim = max(max_sim, sim)
            
            # For novelty: min similarity (inverted later)
            min_sim = min(min_sim, sim)
            
            # For challenge: max similarity in moderate range
            if MODERATE_SIMILARITY_MIN < sim < MODERATE_SIMILARITY_MAX:
                max_moderate_sim = max(max_moderate_sim, sim)
        
        # Reinforcement: how much this reinforces existing identity
        reinforcement = max_sim if self._identity_dimensions else 0.0
        
        # Novelty: inversely related to max similarity
        novelty = max(0.0, 1.0 - max_sim)
        
        # Challenge: high surprise + moderate similarity
        challenge = 0.0
        if max_moderate_sim > 0:
            surprise = float(self.kde.surprise(emb))
            if surprise > 0:
                challenge = min(1.0, surprise * 0.3 * max_moderate_sim)
        
        return reinforcement, challenge, novelty

    def _get_relationship_importance(self, relationship_entity: str) -> float:
        """
        Get the importance of a relationship based on history.
        
        Args:
            relationship_entity: The entity to check importance for
            
        Returns:
            Importance score between 0 and 1
        """
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
        count_factor = min(1.0, math.log1p(interaction_count) / INTERACTION_COUNT_LOG_NORMALIZER)
        
        return 0.3 * count_factor + 0.7 * health

    # -------------------------------------------------------------------------
    # Relational Context Extraction
    # -------------------------------------------------------------------------

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
        
        Args:
            text: Interaction text
            relationship_entity: The identified relationship entity
            actors: All actors in the interaction
            identity_relevance: Previously computed identity relevance score
            
        Returns:
            RelationalContext describing the relational meaning
        """
        # Determine my role based on interaction content AND relationship history
        my_role = self._infer_my_role(text, relationship_entity)
        
        # Estimate relationship quality impact
        quality_delta = self._compute_quality_delta(text, identity_relevance)
        
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
        
        Args:
            text: Interaction text
            relationship_entity: Optional relationship entity for context
            
        Returns:
            Inferred role string
        """
        # Check if we have relationship history
        if relationship_entity:
            story = self.get_relationship_story(relationship_entity)
            if story and story.my_identity_in_this_relationship:
                # Use established identity if role consistency is high
                if story.get_role_consistency(window=5) > ROLE_CONSISTENCY_THRESHOLD:
                    return story.my_identity_in_this_relationship
        
        # Fallback to keyword-based inference
        return self._keyword_based_role(text)
    
    def _keyword_based_role(self, text: str) -> str:
        """
        Keyword-based role inference.
        
        Args:
            text: Interaction text
            
        Returns:
            Inferred role string based on keywords
        """
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

    def _compute_quality_delta(self, text: str, identity_relevance: float) -> float:
        """
        Estimate how this interaction affects relationship quality.
        
        Args:
            text: Interaction text
            identity_relevance: Identity relevance score
            
        Returns:
            Quality delta between -MAX_QUALITY_DELTA and MAX_QUALITY_DELTA
        """
        text_lower = text.lower()
        
        positive_indicators = ["谢谢", "感谢", "太好了", "perfect", "great", "thanks", "helpful", "好的"]
        negative_indicators = ["不对", "错误", "不行", "wrong", "error", "fail", "不满意"]
        
        positive_count = sum(1 for kw in positive_indicators if kw in text_lower)
        negative_count = sum(1 for kw in negative_indicators if kw in text_lower)
        
        # Base delta scaled by identity relevance
        base_delta = (positive_count - negative_count) * QUALITY_DELTA_COEFFICIENT
        return max(-MAX_QUALITY_DELTA, min(MAX_QUALITY_DELTA, base_delta * (0.5 + 0.5 * identity_relevance)))

    def _generate_relational_meaning(self, text: str, my_role: str, quality_delta: float) -> str:
        """
        Generate a brief description of relational meaning.
        
        Args:
            text: Interaction text
            my_role: Inferred role in this interaction
            quality_delta: Impact on relationship quality
            
        Returns:
            Description of relational meaning
        """
        if quality_delta > 0.1:
            return f"作为{my_role}，我们的关系更进一步了"
        elif quality_delta < -0.1:
            return f"作为{my_role}，这次互动有些挑战"
        else:
            return f"作为{my_role}，这是一次常规互动"

    # -------------------------------------------------------------------------
    # Identity Impact Extraction
    # -------------------------------------------------------------------------

    def _extract_identity_impact(
        self, 
        text: str, 
        relational: RelationalContext,
        identity_relevance: float
    ) -> Optional[IdentityImpact]:
        """
        Extract how this interaction impacts my identity.
        
        Key insight: The meaning of an experience can evolve over time.
        
        Args:
            text: Interaction text
            relational: Relational context
            identity_relevance: Identity relevance score
            
        Returns:
            IdentityImpact if significant, None otherwise
        """
        if identity_relevance < IDENTITY_RELEVANCE_THRESHOLD:
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
        """
        Identify which identity dimensions are affected by this interaction.
        
        Args:
            text: Interaction text
            relational: Relational context
            
        Returns:
            List of affected dimension names
        """
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
        
        return dimensions[:MAX_IDENTITY_DIMENSIONS]

    def _generate_identity_meaning(
        self, 
        text: str, 
        relational: RelationalContext,
        affected_dimensions: List[str]
    ) -> str:
        """
        Generate the identity meaning of this interaction.
        
        Args:
            text: Interaction text
            relational: Relational context
            affected_dimensions: List of affected identity dimensions
            
        Returns:
            Description of identity meaning
        """
        dims_str = "、".join(affected_dimensions[:2]) if affected_dimensions else "我的身份"
        
        if relational.relationship_quality_delta > 0.1:
            return f"这次互动强化了{dims_str}"
        elif relational.relationship_quality_delta < -0.1:
            return f"这次互动挑战了{dims_str}"
        else:
            return f"这是{dims_str}的一次体现"

    # -------------------------------------------------------------------------
    # Relationship Story Management
    # -------------------------------------------------------------------------

    def _get_or_create_relationship_story(self, relationship_entity: str) -> StoryArc:
        """
        Get or create a story for a relationship.
        
        Args:
            relationship_entity: The entity the relationship is with
            
        Returns:
            Existing or newly created StoryArc for this relationship
        """
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

    def _update_identity_dimensions(self, plot: Plot) -> None:
        """
        Update identity dimensions based on a plot's identity impact.
        
        Args:
            plot: The plot to extract identity impact from
        """
        if plot.identity_impact:
            for dim in plot.identity_impact.identity_dimensions_affected:
                current = self._identity_dimensions.get(dim, 0.0)
                # Gradual strengthening of identity dimensions
                self._identity_dimensions[dim] = current + IDENTITY_DIMENSION_GROWTH_RATE * (1.0 - current)
