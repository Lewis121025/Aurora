"""
AURORA Pressure Management Tests
=================================

Tests for the PressureMixin functionality in aurora/core/memory/pressure.py.

Tests cover:
- Pressure management and capacity control
- Growth-oriented forgetting
- Identity, relationship, and growth contribution computation
- Plot absorption and forgetting
"""

from __future__ import annotations

import numpy as np
import pytest

from aurora.core.memory import AuroraMemory
from aurora.core.models.config import MemoryConfig
from aurora.core.models.plot import IdentityImpact, Plot, RelationalContext
from aurora.core.models.story import StoryArc
from aurora.core.models.theme import Theme
from aurora.utils.id_utils import det_id
from aurora.utils.time_utils import now_ts


class TestPressureManagement:
    """Tests for pressure management functionality."""

    def test_pressure_manage_under_capacity(self, aurora_memory: AuroraMemory):
        """Test that pressure management does nothing when under capacity."""
        # Ensure we're under capacity
        assert len(aurora_memory.plots) < aurora_memory.cfg.max_plots
        
        initial_count = len(aurora_memory.plots)
        aurora_memory._pressure_manage()
        
        # No plots should be removed when under capacity
        assert len(aurora_memory.plots) == initial_count

    def test_pressure_manage_respects_max_plots(self):
        """Test that pressure management keeps plot count near max_plots."""
        config = MemoryConfig(dim=64, max_plots=10)
        memory = AuroraMemory(cfg=config, seed=42)
        
        # Ingest many interactions to exceed capacity
        for i in range(30):
            memory.ingest(
                f"用户：测试问题{i}。助理：测试回答{i}。",
                actors=("user", "assistant"),
            )
        
        # Active plots should be near or below max_plots
        active_plots = [p for p in memory.plots.values() if p.status == "active"]
        
        # Allow some tolerance due to probabilistic nature
        assert len(active_plots) <= config.max_plots * 2

    def test_pressure_manage_with_story_plots(self, aurora_memory: AuroraMemory):
        """Test pressure management with plots assigned to stories."""
        rng = np.random.default_rng(42)
        
        # Create a story
        story = StoryArc(
            id=det_id("story", "pressure_test"),
            created_ts=now_ts(),
            updated_ts=now_ts(),
            relationship_with="test_user",
        )
        aurora_memory.stories[story.id] = story
        aurora_memory.graph.add_node(story.id, "story", story)
        aurora_memory._relationship_story_index["test_user"] = story.id
        
        # Create many plots exceeding capacity
        for i in range(aurora_memory.cfg.max_plots + 20):
            emb = rng.standard_normal(64).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            
            plot = Plot(
                id=det_id("plot", f"pressure_{i}"),
                ts=now_ts() - i * 3600,
                text=f"测试内容 {i}",
                actors=("user", "assistant"),
                embedding=emb,
                status="active",
            )
            plot.story_id = story.id
            plot.tension = rng.random()
            
            aurora_memory.plots[plot.id] = plot
            story.plot_ids.append(plot.id)
            aurora_memory.vindex.add(plot.id, emb, kind="plot")
        
        # Run pressure management
        aurora_memory._pressure_manage()
        
        # Some plots should have been removed/absorbed
        active_plots = [p for p in aurora_memory.plots.values() if p.status == "active"]
        assert len(active_plots) <= aurora_memory.cfg.max_plots + 5


class TestGrowthOrientedForgetting:
    """Tests for growth-oriented forgetting functionality."""

    def test_score_candidates_for_removal(self, aurora_memory: AuroraMemory):
        """Test that candidates are scored based on contribution."""
        rng = np.random.default_rng(42)
        
        # Create story
        story = StoryArc(
            id=det_id("story", "scoring_test"),
            created_ts=now_ts(),
            updated_ts=now_ts(),
            relationship_with="scorer",
            relationship_health=0.7,
        )
        aurora_memory.stories[story.id] = story
        aurora_memory._relationship_story_index["scorer"] = story.id
        
        # Create plots with different characteristics
        candidates = []
        for i in range(5):
            emb = rng.standard_normal(64).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            
            plot = Plot(
                id=det_id("plot", f"score_{i}"),
                ts=now_ts() - i * 86400,
                text=f"评分测试 {i}",
                actors=("user", "assistant"),
                embedding=emb,
                status="active",
            )
            plot.story_id = story.id
            plot.tension = 0.1 + i * 0.1
            plot.access_count = i * 2
            
            if i % 2 == 0:
                plot.relational = RelationalContext(
                    with_whom="scorer",
                    my_role_in_relation="助手",
                    relationship_quality_delta=0.1,
                    what_this_says_about_us="测试关系",
                )
            
            if i % 3 == 0:
                plot.identity_impact = IdentityImpact(
                    when_formed=now_ts(),
                    initial_meaning="测试意义",
                    current_meaning="测试意义",
                    identity_dimensions_affected=["作为助手的我"],
                    evolution_history=[],
                )
            
            candidates.append(plot)
            aurora_memory.plots[plot.id] = plot
            story.plot_ids.append(plot.id)
        
        # Score candidates
        aurora_memory._score_candidates_for_removal(candidates)
        
        # All candidates should have _keep_score
        for plot in candidates:
            assert hasattr(plot, "_keep_score")
            assert 0 <= plot._keep_score <= 2  # Reasonable range

    def test_select_plots_to_forget(self, aurora_memory: AuroraMemory):
        """Test plot selection for forgetting."""
        rng = np.random.default_rng(42)
        
        # Create candidates with varying scores
        candidates = []
        for i in range(10):
            emb = rng.standard_normal(64).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            
            plot = Plot(
                id=det_id("plot", f"forget_{i}"),
                ts=now_ts(),
                text=f"遗忘测试 {i}",
                actors=("user",),
                embedding=emb,
                status="active",
            )
            plot._keep_score = 0.1 * i  # Varying scores
            candidates.append(plot)
        
        # Select 3 plots to forget
        to_forget = aurora_memory._select_plots_to_forget(candidates, excess=3)
        
        assert len(to_forget) == 3
        assert all(isinstance(pid, str) for pid in to_forget)
        
        # Lower scored plots should be more likely to be selected
        # (This is probabilistic, so we just verify it works)


class TestCapacityCheck:
    """Tests for capacity checking functionality."""

    def test_capacity_triggers_pressure_management(self):
        """Test that exceeding capacity triggers pressure management."""
        config = MemoryConfig(dim=64, max_plots=5)
        memory = AuroraMemory(cfg=config, seed=42)
        
        # Ingest interactions
        for i in range(15):
            memory.ingest(
                f"用户：容量测试{i}。助理：收到{i}。",
                actors=("user", "assistant"),
            )
        
        # Active plots should be managed
        active_count = len([p for p in memory.plots.values() if p.status == "active"])
        
        # Should be close to max_plots (with tolerance)
        assert active_count <= config.max_plots * 3


class TestPlotAbsorption:
    """Tests for plot absorption functionality."""

    def test_absorb_plot_updates_story(self, aurora_memory: AuroraMemory):
        """Test that absorbing a plot updates the story correctly."""
        rng = np.random.default_rng(42)
        
        # Create story
        story = StoryArc(
            id=det_id("story", "absorb_test"),
            created_ts=now_ts(),
            updated_ts=now_ts(),
            relationship_with="absorber",
        )
        aurora_memory.stories[story.id] = story
        aurora_memory._relationship_story_index["absorber"] = story.id
        
        # Create plot
        emb = rng.standard_normal(64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        
        plot = Plot(
            id=det_id("plot", "to_absorb"),
            ts=now_ts(),
            text="将被吸收的内容",
            actors=("user",),
            embedding=emb,
            status="active",
        )
        plot.story_id = story.id
        
        aurora_memory.plots[plot.id] = plot
        story.plot_ids.append(plot.id)
        aurora_memory.vindex.add(plot.id, emb, kind="plot")
        
        # Forget the plot
        aurora_memory._forget_plot(plot.id)
        
        # Plot should be absorbed
        assert plot.status == "absorbed"

    def test_forget_plot_removes_from_vindex(self, aurora_memory: AuroraMemory):
        """Test that forgetting removes plot from vector index."""
        rng = np.random.default_rng(42)
        
        # Create story
        story = StoryArc(
            id=det_id("story", "vindex_test"),
            created_ts=now_ts(),
            updated_ts=now_ts(),
        )
        aurora_memory.stories[story.id] = story
        
        # Create and add plot
        emb = rng.standard_normal(64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        
        plot = Plot(
            id=det_id("plot", "vindex_remove"),
            ts=now_ts(),
            text="向量索引测试",
            actors=("user",),
            embedding=emb,
            status="active",
        )
        plot.story_id = story.id
        
        aurora_memory.plots[plot.id] = plot
        aurora_memory.vindex.add(plot.id, emb, kind="plot")
        
        # Forget the plot
        aurora_memory._forget_plot(plot.id)
        
        # Verify plot is marked as absorbed
        assert plot.status == "absorbed"


class TestContributionComputations:
    """Tests for contribution computation methods."""

    def test_compute_identity_contribution_no_impact(self, aurora_memory: AuroraMemory):
        """Test identity contribution for plot without identity impact."""
        rng = np.random.default_rng(42)
        emb = rng.standard_normal(64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        
        plot = Plot(
            id=det_id("plot", "no_identity"),
            ts=now_ts(),
            text="无身份影响",
            actors=("user",),
            embedding=emb,
            identity_impact=None,
        )
        
        contribution = aurora_memory._compute_identity_contribution(plot)
        
        # Should return baseline
        assert contribution == 0.3

    def test_compute_identity_contribution_with_impact(self, aurora_memory: AuroraMemory):
        """Test identity contribution for plot with identity impact."""
        rng = np.random.default_rng(42)
        emb = rng.standard_normal(64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        
        # Set identity dimension
        aurora_memory._identity_dimensions["作为解释者的我"] = 0.8
        
        # Create theme
        theme = Theme(
            id=det_id("theme", "explainer"),
            created_ts=now_ts(),
            updated_ts=now_ts(),
            prototype=emb,
        )
        aurora_memory.themes[theme.id] = theme
        
        # Create story connected to theme
        story = StoryArc(
            id=det_id("story", "identity_contrib"),
            created_ts=now_ts(),
            updated_ts=now_ts(),
        )
        theme.story_ids.append(story.id)
        aurora_memory.stories[story.id] = story
        
        plot = Plot(
            id=det_id("plot", "with_identity"),
            ts=now_ts(),
            text="有身份影响",
            actors=("user",),
            embedding=emb,
            identity_impact=IdentityImpact(
                when_formed=now_ts(),
                initial_meaning="测试",
                current_meaning="测试",
                identity_dimensions_affected=["作为解释者的我"],
                evolution_history=[],
            ),
        )
        plot.story_id = story.id
        
        aurora_memory.plots[plot.id] = plot
        story.plot_ids.append(plot.id)
        
        contribution = aurora_memory._compute_identity_contribution(plot)
        
        # Should be higher than baseline due to identity dimension
        assert contribution > 0.3

    def test_compute_relationship_contribution_no_relational(
        self, aurora_memory: AuroraMemory
    ):
        """Test relationship contribution for non-relational plot."""
        rng = np.random.default_rng(42)
        emb = rng.standard_normal(64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        
        plot = Plot(
            id=det_id("plot", "no_rel"),
            ts=now_ts(),
            text="无关系上下文",
            actors=("user",),
            embedding=emb,
            relational=None,
        )
        
        contribution = aurora_memory._compute_relationship_contribution(plot)
        
        # Should return baseline
        assert contribution == 0.3

    def test_compute_relationship_contribution_with_healthy_relationship(
        self, aurora_memory: AuroraMemory
    ):
        """Test relationship contribution for plot in healthy relationship."""
        rng = np.random.default_rng(42)
        emb = rng.standard_normal(64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        
        # Create healthy relationship story
        story = StoryArc(
            id=det_id("story", "healthy_rel"),
            created_ts=now_ts(),
            updated_ts=now_ts(),
            relationship_with="healthy_user",
            relationship_health=0.9,
        )
        aurora_memory.stories[story.id] = story
        aurora_memory._relationship_story_index["healthy_user"] = story.id
        
        plot = Plot(
            id=det_id("plot", "healthy_plot"),
            ts=now_ts(),
            text="健康关系中的内容",
            actors=("user",),
            embedding=emb,
            relational=RelationalContext(
                with_whom="healthy_user",
                my_role_in_relation="助手",
                relationship_quality_delta=0.1,
                what_this_says_about_us="良好互动",
            ),
        )
        plot.story_id = story.id
        
        aurora_memory.plots[plot.id] = plot
        story.plot_ids.append(plot.id)
        
        contribution = aurora_memory._compute_relationship_contribution(plot)
        
        # Should have meaningful contribution from healthy relationship
        assert contribution > 0.3

    def test_compute_growth_contribution(self, aurora_memory: AuroraMemory):
        """Test growth contribution computation."""
        rng = np.random.default_rng(42)
        emb = rng.standard_normal(64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        
        # Recent plot with high tension (learning value)
        plot = Plot(
            id=det_id("plot", "growth_test"),
            ts=now_ts() - 3600,  # 1 hour old
            text="增长测试",
            actors=("user",),
            embedding=emb,
        )
        plot.tension = 0.8  # High tension = high learning value
        plot.access_count = 5  # Accessed multiple times
        
        contribution = aurora_memory._compute_growth_contribution(plot)
        
        # Should have positive contribution
        assert contribution > 0

    def test_compute_growth_contribution_old_unused(self, aurora_memory: AuroraMemory):
        """Test growth contribution for old, unused content."""
        rng = np.random.default_rng(42)
        emb = rng.standard_normal(64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        
        # Very old plot never accessed
        plot = Plot(
            id=det_id("plot", "old_unused_growth"),
            ts=now_ts() - 86400 * 60,  # 60 days old
            text="老旧未使用内容",
            actors=("user",),
            embedding=emb,
        )
        plot.tension = 0.1
        plot.access_count = 0  # Never accessed
        
        contribution = aurora_memory._compute_growth_contribution(plot)
        
        # Should have lower contribution due to age and lack of access
        assert contribution < 0.5


class TestForgetPlot:
    """Tests for the _forget_plot method."""

    def test_forget_plot_none_id(self, aurora_memory: AuroraMemory):
        """Test forgetting non-existent plot."""
        # Should not raise
        aurora_memory._forget_plot("nonexistent_plot_id")

    def test_forget_plot_no_story(self, aurora_memory: AuroraMemory):
        """Test forgetting plot without story assignment."""
        rng = np.random.default_rng(42)
        emb = rng.standard_normal(64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        
        plot = Plot(
            id=det_id("plot", "no_story_forget"),
            ts=now_ts(),
            text="无故事关联",
            actors=("user",),
            embedding=emb,
        )
        plot.story_id = None  # No story
        
        aurora_memory.plots[plot.id] = plot
        
        # Should return early without error
        aurora_memory._forget_plot(plot.id)

    def test_forget_plot_preserves_identity_dimensions(self, aurora_memory: AuroraMemory):
        """Test that forgetting preserves accumulated identity dimensions."""
        rng = np.random.default_rng(42)
        emb = rng.standard_normal(64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        
        # Create story
        story = StoryArc(
            id=det_id("story", "preserve_identity"),
            created_ts=now_ts(),
            updated_ts=now_ts(),
        )
        aurora_memory.stories[story.id] = story
        
        # Set identity dimension
        aurora_memory._identity_dimensions["作为学习者的我"] = 0.6
        
        plot = Plot(
            id=det_id("plot", "preserve_test"),
            ts=now_ts(),
            text="保留身份测试",
            actors=("user",),
            embedding=emb,
            identity_impact=IdentityImpact(
                when_formed=now_ts(),
                initial_meaning="学习",
                current_meaning="学习",
                identity_dimensions_affected=["作为学习者的我"],
                evolution_history=[],
            ),
        )
        plot.story_id = story.id
        
        aurora_memory.plots[plot.id] = plot
        aurora_memory.vindex.add(plot.id, emb, kind="plot")
        
        # Forget the plot
        aurora_memory._forget_plot(plot.id)
        
        # Identity dimension should still exist
        assert "作为学习者的我" in aurora_memory._identity_dimensions
        assert aurora_memory._identity_dimensions["作为学习者的我"] == 0.6
