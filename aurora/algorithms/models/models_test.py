"""
AURORA Data Models Tests
========================

Tests for the core data models in aurora.algorithms.models.

Tests cover:
- Plot: Atomic memory unit with relational and identity layers
- StoryArc: Mesoscale relationship narrative
- Theme: Identity dimension / macroscale pattern
- RetrievalTrace, EvolutionSnapshot: Trace and snapshot structures
"""

from __future__ import annotations

import numpy as np
import pytest

from aurora.algorithms.models.plot import Plot, RelationalContext, IdentityImpact
from aurora.algorithms.models.story import StoryArc, RelationshipMoment
from aurora.algorithms.models.theme import Theme
from aurora.algorithms.models.trace import (
    RetrievalTrace,
    EvolutionSnapshot,
    QueryHit,
    EvolutionPatch,
)
from aurora.utils.time_utils import now_ts
from aurora.utils.id_utils import det_id


# =============================================================================
# RelationalContext Tests
# =============================================================================

class TestRelationalContext:
    """Tests for RelationalContext dataclass."""
    
    def test_create_relational_context(self):
        """Test basic creation of RelationalContext."""
        ctx = RelationalContext(
            with_whom="user_001",
            my_role_in_relation="helpful assistant",
            relationship_quality_delta=0.2,
            what_this_says_about_us="We have a good working relationship",
        )
        
        assert ctx.with_whom == "user_001"
        assert ctx.my_role_in_relation == "helpful assistant"
        assert ctx.relationship_quality_delta == 0.2
        assert ctx.what_this_says_about_us == "We have a good working relationship"
    
    def test_relational_context_round_trip(self):
        """Test serialization round-trip preserves data."""
        ctx = RelationalContext(
            with_whom="user_002",
            my_role_in_relation="mentor",
            relationship_quality_delta=-0.1,
            what_this_says_about_us="学习过程中的挫折",
        )
        
        state = ctx.to_state_dict()
        restored = RelationalContext.from_state_dict(state)
        
        assert restored.with_whom == ctx.with_whom
        assert restored.my_role_in_relation == ctx.my_role_in_relation
        assert restored.relationship_quality_delta == ctx.relationship_quality_delta
        assert restored.what_this_says_about_us == ctx.what_this_says_about_us
    
    def test_relational_context_backward_compat_alias(self):
        """Test that to_dict/from_dict aliases work."""
        ctx = RelationalContext(
            with_whom="user",
            my_role_in_relation="assistant",
            relationship_quality_delta=0.0,
            what_this_says_about_us="neutral",
        )
        
        # to_dict should be same as to_state_dict
        assert ctx.to_dict() == ctx.to_state_dict()
        
        # from_dict should work same as from_state_dict
        state = ctx.to_dict()
        restored = RelationalContext.from_dict(state)
        assert restored.with_whom == ctx.with_whom


# =============================================================================
# IdentityImpact Tests
# =============================================================================

class TestIdentityImpact:
    """Tests for IdentityImpact dataclass."""
    
    def test_create_identity_impact(self):
        """Test basic creation of IdentityImpact."""
        ts = now_ts()
        impact = IdentityImpact(
            when_formed=ts,
            initial_meaning="This shows my capability as an explainer",
            current_meaning="This shows my capability as an explainer",
            identity_dimensions_affected=["explainer", "helper"],
            evolution_history=[],
        )
        
        assert impact.when_formed == ts
        assert impact.initial_meaning == impact.current_meaning
        assert len(impact.identity_dimensions_affected) == 2
        assert len(impact.evolution_history) == 0
    
    def test_identity_impact_update_meaning(self):
        """Test that update_meaning records evolution history."""
        ts = now_ts()
        impact = IdentityImpact(
            when_formed=ts,
            initial_meaning="A failure",
            current_meaning="A failure",
            identity_dimensions_affected=["resilience"],
            evolution_history=[],
        )
        
        # Update meaning
        impact.update_meaning("A learning experience")
        
        assert impact.current_meaning == "A learning experience"
        assert impact.initial_meaning == "A failure"  # Initial unchanged
        assert len(impact.evolution_history) == 1
        assert impact.evolution_history[0][1] == "A failure"  # Old meaning in history
    
    def test_identity_impact_no_update_when_same(self):
        """Test that update_meaning does not record history when meaning unchanged."""
        ts = now_ts()
        impact = IdentityImpact(
            when_formed=ts,
            initial_meaning="Same meaning",
            current_meaning="Same meaning",
            identity_dimensions_affected=[],
            evolution_history=[],
        )
        
        impact.update_meaning("Same meaning")
        
        assert len(impact.evolution_history) == 0
    
    def test_identity_impact_round_trip(self):
        """Test serialization round-trip preserves data."""
        ts = now_ts()
        impact = IdentityImpact(
            when_formed=ts,
            initial_meaning="Initial",
            current_meaning="Evolved",
            identity_dimensions_affected=["dim1", "dim2"],
            evolution_history=[(ts - 100, "Old meaning")],
        )
        
        state = impact.to_state_dict()
        restored = IdentityImpact.from_state_dict(state)
        
        assert restored.when_formed == impact.when_formed
        assert restored.initial_meaning == impact.initial_meaning
        assert restored.current_meaning == impact.current_meaning
        assert restored.identity_dimensions_affected == impact.identity_dimensions_affected
        assert len(restored.evolution_history) == 1


# =============================================================================
# Plot Tests
# =============================================================================

class TestPlot:
    """Tests for Plot dataclass."""
    
    @pytest.fixture
    def sample_embedding(self) -> np.ndarray:
        """Create a normalized sample embedding."""
        rng = np.random.default_rng(42)
        emb = rng.standard_normal(64).astype(np.float32)
        return emb / np.linalg.norm(emb)
    
    @pytest.fixture
    def basic_plot(self, sample_embedding: np.ndarray) -> Plot:
        """Create a basic plot without relational/identity layers."""
        return Plot(
            id=det_id("plot", "basic_test"),
            ts=now_ts(),
            text="用户：测试交互。助理：收到。",
            actors=("user", "assistant"),
            embedding=sample_embedding,
        )
    
    @pytest.fixture
    def full_plot(self, sample_embedding: np.ndarray) -> Plot:
        """Create a plot with all layers populated."""
        ts = now_ts()
        return Plot(
            id=det_id("plot", "full_test"),
            ts=ts,
            text="用户：帮我解释递归。助理：递归是函数调用自身的过程。",
            actors=("user", "assistant"),
            embedding=sample_embedding,
            surprise=0.5,
            pred_error=0.3,
            redundancy=0.1,
            goal_relevance=0.8,
            tension=0.4,
            relational=RelationalContext(
                with_whom="user",
                my_role_in_relation="teacher",
                relationship_quality_delta=0.2,
                what_this_says_about_us="良好的学习关系",
            ),
            identity_impact=IdentityImpact(
                when_formed=ts,
                initial_meaning="展现了我作为解释者的能力",
                current_meaning="展现了我作为解释者的能力",
                identity_dimensions_affected=["作为解释者的我"],
                evolution_history=[],
            ),
            story_id="story_001",
            access_count=5,
        )
    
    def test_plot_creation(self, basic_plot: Plot):
        """Test basic plot creation."""
        assert basic_plot.id is not None
        assert basic_plot.ts > 0
        assert basic_plot.text.startswith("用户")
        assert basic_plot.actors == ("user", "assistant")
        assert basic_plot.embedding.shape == (64,)
        assert basic_plot.status == "active"
    
    def test_plot_has_identity_impact_false(self, basic_plot: Plot):
        """Test has_identity_impact returns False when no impact."""
        assert basic_plot.has_identity_impact() is False
    
    def test_plot_has_identity_impact_true(self, full_plot: Plot):
        """Test has_identity_impact returns True when impact present."""
        assert full_plot.has_identity_impact() is True
    
    def test_plot_get_relationship_entity_none(self, basic_plot: Plot):
        """Test get_relationship_entity returns None when no relational context."""
        assert basic_plot.get_relationship_entity() is None
    
    def test_plot_get_relationship_entity(self, full_plot: Plot):
        """Test get_relationship_entity returns correct entity."""
        assert full_plot.get_relationship_entity() == "user"
    
    def test_plot_get_my_role_default(self, basic_plot: Plot):
        """Test get_my_role returns default when no relational context."""
        assert basic_plot.get_my_role() == "assistant"
    
    def test_plot_get_my_role(self, full_plot: Plot):
        """Test get_my_role returns correct role."""
        assert full_plot.get_my_role() == "teacher"
    
    def test_plot_get_identity_dimensions(self, full_plot: Plot):
        """Test get_identity_dimensions returns affected dimensions."""
        dims = full_plot.get_identity_dimensions()
        assert "作为解释者的我" in dims
    
    def test_plot_get_identity_dimensions_empty(self, basic_plot: Plot):
        """Test get_identity_dimensions returns empty list when no impact."""
        dims = basic_plot.get_identity_dimensions()
        assert dims == []
    
    def test_plot_mass_positive(self, full_plot: Plot):
        """Test mass computation returns positive value."""
        mass = full_plot.mass()
        assert mass > 0
    
    def test_plot_round_trip_basic(self, basic_plot: Plot):
        """Test serialization round-trip for basic plot."""
        state = basic_plot.to_state_dict()
        restored = Plot.from_state_dict(state)
        
        assert restored.id == basic_plot.id
        assert restored.ts == basic_plot.ts
        assert restored.text == basic_plot.text
        assert restored.actors == basic_plot.actors
        assert np.allclose(restored.embedding, basic_plot.embedding)
        assert restored.status == basic_plot.status
    
    def test_plot_round_trip_full(self, full_plot: Plot):
        """Test serialization round-trip for full plot with all layers."""
        state = full_plot.to_state_dict()
        restored = Plot.from_state_dict(state)
        
        assert restored.id == full_plot.id
        assert np.allclose(restored.embedding, full_plot.embedding)
        
        # Check relational context
        assert restored.relational is not None
        assert restored.relational.with_whom == full_plot.relational.with_whom
        assert restored.relational.my_role_in_relation == full_plot.relational.my_role_in_relation
        
        # Check identity impact
        assert restored.identity_impact is not None
        assert restored.identity_impact.initial_meaning == full_plot.identity_impact.initial_meaning
        
        # Check signals
        assert restored.surprise == full_plot.surprise
        assert restored.tension == full_plot.tension
        assert restored.story_id == full_plot.story_id
        assert restored.access_count == full_plot.access_count


# =============================================================================
# RelationshipMoment Tests
# =============================================================================

class TestRelationshipMoment:
    """Tests for RelationshipMoment dataclass."""
    
    def test_create_relationship_moment(self):
        """Test basic creation of RelationshipMoment."""
        ts = now_ts()
        moment = RelationshipMoment(
            ts=ts,
            event_summary="用户请求帮助，我成功提供了解答",
            trust_level=0.7,
            my_role="helper",
            quality_delta=0.1,
        )
        
        assert moment.ts == ts
        assert moment.trust_level == 0.7
        assert moment.my_role == "helper"
        assert moment.quality_delta == 0.1
    
    def test_relationship_moment_round_trip(self):
        """Test serialization round-trip."""
        ts = now_ts()
        moment = RelationshipMoment(
            ts=ts,
            event_summary="测试事件",
            trust_level=0.85,
            my_role="assistant",
            quality_delta=-0.05,
        )
        
        state = moment.to_state_dict()
        restored = RelationshipMoment.from_state_dict(state)
        
        assert restored.ts == moment.ts
        assert restored.event_summary == moment.event_summary
        assert restored.trust_level == moment.trust_level
        assert restored.my_role == moment.my_role
        assert restored.quality_delta == moment.quality_delta


# =============================================================================
# StoryArc Tests
# =============================================================================

class TestStoryArc:
    """Tests for StoryArc dataclass."""
    
    @pytest.fixture
    def sample_centroid(self) -> np.ndarray:
        """Create a sample centroid embedding."""
        rng = np.random.default_rng(42)
        emb = rng.standard_normal(64).astype(np.float32)
        return emb / np.linalg.norm(emb)
    
    @pytest.fixture
    def basic_story(self, sample_centroid: np.ndarray) -> StoryArc:
        """Create a basic story."""
        ts = now_ts()
        return StoryArc(
            id=det_id("story", "basic"),
            created_ts=ts,
            updated_ts=ts,
            centroid=sample_centroid,
        )
    
    @pytest.fixture
    def relationship_story(self, sample_centroid: np.ndarray) -> StoryArc:
        """Create a relationship-centric story."""
        ts = now_ts()
        story = StoryArc(
            id=det_id("story", "rel_user"),
            created_ts=ts,
            updated_ts=ts,
            relationship_with="user_001",
            relationship_type="user",
            my_identity_in_this_relationship="helpful mentor",
            centroid=sample_centroid,
        )
        # Add some relationship moments
        story.add_relationship_moment(
            event_summary="First meeting",
            trust_level=0.5,
            my_role="assistant",
            quality_delta=0.0,
            ts=ts - 1000,
        )
        story.add_relationship_moment(
            event_summary="Helped with difficult problem",
            trust_level=0.7,
            my_role="mentor",
            quality_delta=0.1,
            ts=ts - 500,
        )
        return story
    
    def test_story_creation(self, basic_story: StoryArc):
        """Test basic story creation."""
        assert basic_story.id is not None
        assert basic_story.created_ts > 0
        assert basic_story.status == "developing"
        assert basic_story.plot_ids == []
    
    def test_story_is_relationship_story_false(self, basic_story: StoryArc):
        """Test is_relationship_story returns False for non-relationship story."""
        assert basic_story.is_relationship_story() is False
    
    def test_story_is_relationship_story_true(self, relationship_story: StoryArc):
        """Test is_relationship_story returns True for relationship story."""
        assert relationship_story.is_relationship_story() is True
    
    def test_add_relationship_moment(self, relationship_story: StoryArc):
        """Test adding relationship moments."""
        initial_count = len(relationship_story.relationship_arc)
        
        relationship_story.add_relationship_moment(
            event_summary="New interaction",
            trust_level=0.8,
            my_role="advisor",
            quality_delta=0.05,
        )
        
        assert len(relationship_story.relationship_arc) == initial_count + 1
        last_moment = relationship_story.relationship_arc[-1]
        assert last_moment.event_summary == "New interaction"
        assert last_moment.trust_level == 0.8
    
    def test_add_relationship_moment_updates_health(self, basic_story: StoryArc):
        """Test that adding moments updates relationship health."""
        initial_health = basic_story.relationship_health
        
        # Positive quality delta should increase health
        basic_story.add_relationship_moment(
            event_summary="Good interaction",
            trust_level=0.7,
            my_role="assistant",
            quality_delta=0.5,  # Strong positive delta
        )
        
        assert basic_story.relationship_health > initial_health
    
    def test_activity_probability(self, basic_story: StoryArc):
        """Test activity probability is in valid range."""
        prob = basic_story.activity_probability()
        assert 0 < prob <= 1.0
    
    def test_activity_probability_decreases_with_idle(self, basic_story: StoryArc):
        """Test that activity probability decreases as story becomes idle."""
        # Fresh story
        prob_fresh = basic_story.activity_probability()
        
        # Simulate idle by checking probability at future time
        future_ts = basic_story.updated_ts + 86400  # 1 day later
        prob_idle = basic_story.activity_probability(ts=future_ts)
        
        assert prob_idle < prob_fresh
    
    def test_get_current_trust(self, relationship_story: StoryArc):
        """Test get_current_trust returns last trust level."""
        trust = relationship_story.get_current_trust()
        assert trust == 0.7  # Last added moment had trust_level=0.7
    
    def test_get_current_trust_default(self, basic_story: StoryArc):
        """Test get_current_trust returns default when no moments."""
        trust = basic_story.get_current_trust()
        assert trust == 0.5  # Neutral default
    
    def test_get_trust_trend(self, relationship_story: StoryArc):
        """Test trust trend calculation."""
        # Add more moments to test trend
        for i in range(4):
            relationship_story.add_relationship_moment(
                event_summary=f"Interaction {i}",
                trust_level=0.7 + i * 0.05,  # Increasing trust
                my_role="mentor",
            )
        
        trend = relationship_story.get_trust_trend()
        assert trend > 0  # Should be positive (increasing)
    
    def test_add_lesson(self, relationship_story: StoryArc):
        """Test adding lessons to relationship."""
        relationship_story.add_lesson("Patience is key")
        relationship_story.add_lesson("Clear communication helps")
        
        assert "Patience is key" in relationship_story.lessons_from_relationship
        assert len(relationship_story.lessons_from_relationship) == 2
        
        # Adding duplicate should not increase count
        relationship_story.add_lesson("Patience is key")
        assert len(relationship_story.lessons_from_relationship) == 2
    
    def test_story_mass_positive(self, relationship_story: StoryArc):
        """Test mass computation returns positive value."""
        relationship_story.plot_ids = ["plot_1", "plot_2"]
        mass = relationship_story.mass()
        assert mass > 0
    
    def test_update_stats_welford(self, basic_story: StoryArc):
        """Test Welford's algorithm for running statistics."""
        # Add some distance values
        basic_story._update_stats("dist", 1.0)
        basic_story._update_stats("dist", 2.0)
        basic_story._update_stats("dist", 3.0)
        
        assert basic_story.dist_n == 3
        assert basic_story.dist_mean == 2.0  # Mean of 1, 2, 3
        assert basic_story.dist_var() > 0
    
    def test_story_round_trip(self, relationship_story: StoryArc):
        """Test serialization round-trip preserves all data."""
        # Add narrative structure
        relationship_story.setup = "Initial meeting"
        relationship_story.rising_action = ["Problem identified", "Working together"]
        relationship_story.climax = "Breakthrough moment"
        relationship_story.central_conflict = "Complex technical challenge"
        relationship_story.add_turning_point(now_ts(), "User achieved understanding")
        
        state = relationship_story.to_state_dict()
        restored = StoryArc.from_state_dict(state)
        
        assert restored.id == relationship_story.id
        assert restored.relationship_with == relationship_story.relationship_with
        assert restored.my_identity_in_this_relationship == relationship_story.my_identity_in_this_relationship
        assert len(restored.relationship_arc) == len(relationship_story.relationship_arc)
        assert restored.setup == relationship_story.setup
        assert restored.climax == relationship_story.climax
        assert len(restored.turning_points) == len(relationship_story.turning_points)
        
        if relationship_story.centroid is not None:
            assert np.allclose(restored.centroid, relationship_story.centroid)
    
    def test_narrative_phase_detection(self):
        """Test get_narrative_phase returns correct phase."""
        ts = now_ts()
        story = StoryArc(id="test", created_ts=ts, updated_ts=ts)
        
        assert story.get_narrative_phase() == "unknown"
        
        story.setup = "Beginning"
        assert story.get_narrative_phase() == "setup"
        
        story.rising_action = ["Event 1"]
        assert story.get_narrative_phase() == "rising"
        
        story.climax = "Peak moment"
        assert story.get_narrative_phase() == "climax"
        
        story.resolution = "Conclusion"
        assert story.get_narrative_phase() == "resolution"
    
    def test_narrative_completeness(self):
        """Test narrative completeness calculation."""
        ts = now_ts()
        story = StoryArc(id="test", created_ts=ts, updated_ts=ts)
        
        # Empty story
        assert story.get_narrative_completeness() == 0.0
        
        # Add elements
        story.setup = "Beginning"
        story.climax = "Peak"
        story.resolution = "End"
        
        completeness = story.get_narrative_completeness()
        assert 0 < completeness < 1.0  # Partial completeness


# =============================================================================
# Theme Tests
# =============================================================================

class TestTheme:
    """Tests for Theme dataclass."""
    
    @pytest.fixture
    def sample_prototype(self) -> np.ndarray:
        """Create a sample prototype embedding."""
        rng = np.random.default_rng(42)
        emb = rng.standard_normal(64).astype(np.float32)
        return emb / np.linalg.norm(emb)
    
    @pytest.fixture
    def basic_theme(self, sample_prototype: np.ndarray) -> Theme:
        """Create a basic theme."""
        ts = now_ts()
        return Theme(
            id=det_id("theme", "basic"),
            created_ts=ts,
            updated_ts=ts,
            prototype=sample_prototype,
            name="Helper Pattern",
            theme_type="pattern",
        )
    
    @pytest.fixture
    def identity_theme(self, sample_prototype: np.ndarray) -> Theme:
        """Create an identity dimension theme."""
        ts = now_ts()
        return Theme(
            id=det_id("theme", "identity"),
            created_ts=ts,
            updated_ts=ts,
            prototype=sample_prototype,
            identity_dimension="作为解释者的我",
            supporting_relationships=["user_001", "user_002"],
            theme_type="identity",
            a=5.0,  # More successes
            b=2.0,
        )
    
    def test_theme_creation(self, basic_theme: Theme):
        """Test basic theme creation."""
        assert basic_theme.id is not None
        assert basic_theme.created_ts > 0
        assert basic_theme.name == "Helper Pattern"
        assert basic_theme.theme_type == "pattern"
    
    def test_confidence_initial(self, basic_theme: Theme):
        """Test confidence with initial a=1, b=1 is 0.5."""
        confidence = basic_theme.confidence()
        assert confidence == 0.5  # 1 / (1 + 1)
    
    def test_confidence_after_updates(self, basic_theme: Theme):
        """Test confidence after evidence updates."""
        # Add successes
        basic_theme.update_evidence(success=True)
        basic_theme.update_evidence(success=True)
        basic_theme.update_evidence(success=True)
        
        confidence = basic_theme.confidence()
        # a = 4, b = 1, confidence = 4/5 = 0.8
        assert confidence == 4.0 / 5.0
    
    def test_confidence_with_failures(self, basic_theme: Theme):
        """Test confidence decreases with failures."""
        basic_theme.update_evidence(success=False)
        basic_theme.update_evidence(success=False)
        
        confidence = basic_theme.confidence()
        # a = 1, b = 3, confidence = 1/4 = 0.25
        assert confidence == 1.0 / 4.0
    
    def test_is_identity_dimension_false(self, basic_theme: Theme):
        """Test is_identity_dimension returns False for pattern theme."""
        assert basic_theme.is_identity_dimension() is False
    
    def test_is_identity_dimension_true_by_dimension(self, identity_theme: Theme):
        """Test is_identity_dimension returns True when identity_dimension set."""
        assert identity_theme.is_identity_dimension() is True
    
    def test_is_identity_dimension_true_by_type(self, sample_prototype: np.ndarray):
        """Test is_identity_dimension returns True when theme_type is identity."""
        ts = now_ts()
        theme = Theme(
            id="test",
            created_ts=ts,
            updated_ts=ts,
            theme_type="identity",
        )
        assert theme.is_identity_dimension() is True
    
    def test_identity_strength(self, identity_theme: Theme):
        """Test identity strength calculation combines multiple signals."""
        strength = identity_theme.identity_strength()
        assert 0 < strength <= 1.0
        
        # Should be higher than basic theme due to evidence and relationships
        basic = Theme(
            id="basic", created_ts=now_ts(), updated_ts=now_ts(),
            identity_dimension="test",
        )
        assert strength > basic.identity_strength()
    
    def test_add_supporting_relationship(self, basic_theme: Theme):
        """Test adding supporting relationships."""
        basic_theme.identity_dimension = "作为帮助者的我"
        
        basic_theme.add_supporting_relationship("user_001")
        basic_theme.add_supporting_relationship("user_002")
        
        assert "user_001" in basic_theme.supporting_relationships
        assert "user_002" in basic_theme.supporting_relationships
        
        # Strength is recalculated based on formula:
        # 0.5 * confidence + 0.3 * relationship_factor + 0.2 * story_factor
        # With 2 relationships and confidence=0.5, strength should be recalculated
        expected_relationship_factor = min(1.0, 2 * 0.15)  # = 0.3
        expected_story_factor = 0.0  # No stories
        expected_strength = 0.5 * 0.5 + 0.3 * expected_relationship_factor + 0.2 * expected_story_factor
        assert abs(basic_theme.strength - expected_strength) < 0.01
        
        # Adding duplicate should not increase
        basic_theme.add_supporting_relationship("user_001")
        assert len(basic_theme.supporting_relationships) == 2
    
    def test_add_tension(self, identity_theme: Theme):
        """Test adding tensions with other themes."""
        identity_theme.add_tension("theme_strict")
        identity_theme.add_tension("theme_perfectionist")
        
        assert "theme_strict" in identity_theme.tensions_with
        assert len(identity_theme.tensions_with) == 2
        
        # No duplicates
        identity_theme.add_tension("theme_strict")
        assert len(identity_theme.tensions_with) == 2
    
    def test_add_harmony(self, identity_theme: Theme):
        """Test adding harmonies with other themes."""
        identity_theme.add_harmony("theme_patient")
        identity_theme.add_harmony("theme_curious")
        
        assert "theme_patient" in identity_theme.harmonizes_with
        assert len(identity_theme.harmonizes_with) == 2
    
    def test_has_significant_tensions(self, identity_theme: Theme):
        """Test significant tensions detection."""
        assert identity_theme.has_significant_tensions() is False
        
        identity_theme.add_tension("t1")
        identity_theme.add_tension("t2")
        identity_theme.add_tension("t3")
        
        assert identity_theme.has_significant_tensions() is True
    
    def test_theme_mass_positive(self, identity_theme: Theme):
        """Test mass computation returns positive value."""
        identity_theme.story_ids = ["story_1", "story_2"]
        mass = identity_theme.mass()
        assert mass > 0
    
    def test_theme_round_trip(self, identity_theme: Theme):
        """Test serialization round-trip preserves all data."""
        identity_theme.add_tension("theme_conflict")
        identity_theme.add_harmony("theme_complement")
        identity_theme.story_ids = ["story_1", "story_2"]
        
        state = identity_theme.to_state_dict()
        restored = Theme.from_state_dict(state)
        
        assert restored.id == identity_theme.id
        assert restored.identity_dimension == identity_theme.identity_dimension
        assert restored.supporting_relationships == identity_theme.supporting_relationships
        assert restored.tensions_with == identity_theme.tensions_with
        assert restored.harmonizes_with == identity_theme.harmonizes_with
        assert restored.a == identity_theme.a
        assert restored.b == identity_theme.b
        assert restored.theme_type == identity_theme.theme_type
        
        if identity_theme.prototype is not None:
            assert np.allclose(restored.prototype, identity_theme.prototype)
    
    def test_to_identity_narrative(self, identity_theme: Theme):
        """Test generating identity narrative."""
        narrative = identity_theme.to_identity_narrative()
        assert len(narrative) > 0
        assert "解释者" in narrative  # Should mention identity dimension


# =============================================================================
# RetrievalTrace Tests
# =============================================================================

class TestRetrievalTrace:
    """Tests for RetrievalTrace dataclass."""
    
    def test_create_retrieval_trace(self):
        """Test basic creation of RetrievalTrace."""
        query_emb = np.random.randn(64).astype(np.float32)
        attractor_path = [np.random.randn(64).astype(np.float32) for _ in range(3)]
        
        trace = RetrievalTrace(
            query="What did we discuss about recursion?",
            query_emb=query_emb,
            attractor_path=attractor_path,
            ranked=[
                ("plot_1", 0.95, "plot"),
                ("plot_2", 0.87, "plot"),
                ("story_1", 0.82, "story"),
            ],
            asker_id="user_001",
            activated_identity="teacher",
            relationship_context="Teaching relationship",
        )
        
        assert trace.query == "What did we discuss about recursion?"
        assert len(trace.ranked) == 3
        assert trace.asker_id == "user_001"
        assert trace.activated_identity == "teacher"
    
    def test_retrieval_trace_minimal(self):
        """Test RetrievalTrace with minimal required fields."""
        query_emb = np.zeros(64, dtype=np.float32)
        
        trace = RetrievalTrace(
            query="simple query",
            query_emb=query_emb,
            attractor_path=[],
            ranked=[],
        )
        
        assert trace.asker_id is None
        assert trace.activated_identity is None
        assert trace.relationship_context is None


# =============================================================================
# EvolutionSnapshot Tests
# =============================================================================

class TestEvolutionSnapshot:
    """Tests for EvolutionSnapshot dataclass."""
    
    def test_create_evolution_snapshot(self):
        """Test basic creation of EvolutionSnapshot."""
        centroid = np.random.randn(64).astype(np.float32)
        prototype = np.random.randn(64).astype(np.float32)
        
        snapshot = EvolutionSnapshot(
            story_ids=["story_1", "story_2"],
            story_statuses={"story_1": "developing", "story_2": "resolved"},
            story_centroids={"story_1": centroid, "story_2": None},
            story_tension_curves={"story_1": [0.1, 0.3, 0.5], "story_2": []},
            story_updated_ts={"story_1": 1000.0, "story_2": 2000.0},
            story_gap_means={"story_1": 300.0, "story_2": 600.0},
            theme_ids=["theme_1"],
            theme_story_counts={"theme_1": 2},
            theme_prototypes={"theme_1": prototype},
            crp_theme_alpha=0.5,
            rng_state={"bit_generator": "PCG64"},
        )
        
        assert len(snapshot.story_ids) == 2
        assert snapshot.story_statuses["story_1"] == "developing"
        assert len(snapshot.theme_ids) == 1
        assert snapshot.crp_theme_alpha == 0.5


# =============================================================================
# QueryHit Tests
# =============================================================================

class TestQueryHit:
    """Tests for QueryHit dataclass."""
    
    def test_create_query_hit(self):
        """Test basic creation of QueryHit."""
        hit = QueryHit(
            id="plot_001",
            kind="plot",
            score=0.92,
            snippet="用户讨论了递归的概念...",
            metadata={"story_id": "story_001"},
        )
        
        assert hit.id == "plot_001"
        assert hit.kind == "plot"
        assert hit.score == 0.92
        assert hit.snippet.startswith("用户")
        assert hit.metadata["story_id"] == "story_001"
    
    def test_query_hit_without_metadata(self):
        """Test QueryHit without metadata."""
        hit = QueryHit(
            id="theme_001",
            kind="theme",
            score=0.75,
            snippet="Identity dimension: helper",
        )
        
        assert hit.metadata is None


# =============================================================================
# EvolutionPatch Tests
# =============================================================================

class TestEvolutionPatch:
    """Tests for EvolutionPatch dataclass."""
    
    def test_create_evolution_patch(self):
        """Test basic creation of EvolutionPatch."""
        prototype = np.random.randn(64).astype(np.float32)
        
        patch = EvolutionPatch(
            status_changes={"story_1": "resolved", "story_2": "abandoned"},
            theme_assignments=[("story_1", "theme_1"), ("story_3", "theme_2")],
            new_themes=[("theme_new", prototype)],
        )
        
        assert len(patch.status_changes) == 2
        assert patch.status_changes["story_1"] == "resolved"
        assert len(patch.theme_assignments) == 2
        assert len(patch.new_themes) == 1
    
    def test_evolution_patch_empty(self):
        """Test EvolutionPatch with default empty values."""
        patch = EvolutionPatch()
        
        assert patch.status_changes == {}
        assert patch.theme_assignments == []
        assert patch.new_themes == []
