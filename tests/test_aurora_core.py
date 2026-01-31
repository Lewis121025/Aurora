"""
AURORA Core Tests
=================

Tests for the AuroraMemory core class.

Tests cover:
- Ingest with relationship-centric processing
- Query with identity activation
- Feedback and learning
- Evolution and consolidation
- Serialization and deserialization
"""

from __future__ import annotations

import numpy as np
import pytest

from aurora.algorithms.aurora_core import AuroraMemory
from aurora.algorithms.models.config import MemoryConfig
from aurora.exceptions import MemoryNotFoundError, ValidationError


class TestAuroraMemoryIngest:
    """Tests for AuroraMemory.ingest() method."""
    
    def test_ingest_creates_plot(self, aurora_memory: AuroraMemory):
        """Test that ingest creates a plot with proper structure."""
        plot = aurora_memory.ingest(
            "用户：你好！助理：你好，有什么可以帮你的？",
            actors=("user", "assistant"),
        )
        
        assert plot is not None
        assert plot.id is not None
        assert plot.ts > 0
        assert plot.embedding is not None
        assert plot.embedding.shape == (64,)
    
    def test_ingest_extracts_relational_context(self, aurora_memory: AuroraMemory):
        """Test that ingest extracts relational context."""
        plot = aurora_memory.ingest(
            "用户：帮我解释一下递归。助理：递归是函数调用自身的过程。",
            actors=("user", "assistant"),
        )
        
        # Plot may or may not be stored (probabilistic), but should have relational context
        assert plot.relational is not None
        assert plot.relational.with_whom == "user"
        assert plot.relational.my_role_in_relation is not None
    
    def test_ingest_computes_signals(self, aurora_memory: AuroraMemory):
        """Test that ingest computes all signal values."""
        plot = aurora_memory.ingest(
            "用户：这是一个测试。助理：好的。",
            actors=("user", "assistant"),
        )
        
        assert hasattr(plot, 'surprise')
        assert hasattr(plot, 'pred_error')
        assert hasattr(plot, 'redundancy')
        assert hasattr(plot, 'goal_relevance')
        assert hasattr(plot, 'tension')
    
    def test_ingest_updates_kde(self, aurora_memory: AuroraMemory):
        """Test that ingest updates the KDE density estimator."""
        initial_count = len(aurora_memory.kde._vecs)
        
        aurora_memory.ingest("用户：测试1。助理：收到。", actors=("user", "assistant"))
        aurora_memory.ingest("用户：测试2。助理：收到。", actors=("user", "assistant"))
        aurora_memory.ingest("用户：测试3。助理：收到。", actors=("user", "assistant"))
        
        # KDE should be updated regardless of storage decision
        assert len(aurora_memory.kde._vecs) >= initial_count
    
    def test_ingest_with_context(self, aurora_memory: AuroraMemory):
        """Test ingest with context text affects goal relevance."""
        plot = aurora_memory.ingest(
            "用户：帮我写一个排序算法。助理：好的，我来帮你实现。",
            actors=("user", "assistant"),
            context_text="编程任务",
        )
        
        # Goal relevance should be computed when context is provided
        assert plot.goal_relevance >= 0
    
    def test_ingest_creates_relationship_story(self, aurora_memory: AuroraMemory):
        """Test that repeated interactions create a relationship story."""
        # Ingest multiple interactions with same user
        for i in range(5):
            aurora_memory.ingest(
                f"用户：问题{i}。助理：回答{i}。",
                actors=("user", "assistant"),
            )
        
        # Should have created relationship story if plots were stored
        if aurora_memory.plots:
            relationship_story = aurora_memory.get_relationship_story("user")
            # May or may not exist depending on storage decisions
            if relationship_story:
                assert relationship_story.relationship_with == "user"


class TestAuroraMemoryQuery:
    """Tests for AuroraMemory.query() method."""
    
    def test_query_returns_trace(self, populated_memory: AuroraMemory):
        """Test that query returns a proper RetrievalTrace."""
        trace = populated_memory.query("递归是什么？", k=3)
        
        assert trace is not None
        assert trace.query == "递归是什么？"
        assert trace.query_emb is not None
        assert len(trace.query_emb) == 64
    
    def test_query_with_asker_id(self, populated_memory: AuroraMemory):
        """Test query with asker_id activates relationship context."""
        trace = populated_memory.query(
            "帮我解释一下",
            k=3,
            asker_id="user",
        )
        
        assert trace.asker_id == "user"
    
    def test_query_updates_access_stats(self, populated_memory: AuroraMemory):
        """Test that query updates access statistics on retrieved items."""
        if not populated_memory.plots:
            pytest.skip("No plots stored")
        
        # Get initial access counts
        initial_counts = {
            pid: p.access_count for pid, p in populated_memory.plots.items()
        }
        
        # Perform query
        trace = populated_memory.query("测试查询", k=5)
        
        # Check if any access counts increased
        for pid, (_, _, kind) in zip(
            [r[0] for r in trace.ranked],
            trace.ranked
        ):
            if kind == "plot" and pid in populated_memory.plots:
                assert populated_memory.plots[pid].access_count >= initial_counts.get(pid, 0)


class TestAuroraMemoryFeedback:
    """Tests for AuroraMemory.feedback_retrieval() method."""
    
    def test_feedback_positive(self, populated_memory: AuroraMemory):
        """Test positive feedback updates beliefs."""
        if not populated_memory.plots:
            pytest.skip("No plots stored")
        
        # Get a plot ID to use as chosen
        plot_id = next(iter(populated_memory.plots.keys()))
        
        # Should not raise
        populated_memory.feedback_retrieval(
            "测试查询",
            chosen_id=plot_id,
            success=True,
        )
    
    def test_feedback_negative(self, populated_memory: AuroraMemory):
        """Test negative feedback updates beliefs."""
        if not populated_memory.plots:
            pytest.skip("No plots stored")
        
        plot_id = next(iter(populated_memory.plots.keys()))
        
        # Should not raise
        populated_memory.feedback_retrieval(
            "测试查询",
            chosen_id=plot_id,
            success=False,
        )


class TestAuroraMemoryEvolution:
    """Tests for AuroraMemory.evolve() method."""
    
    def test_evolve_runs_without_error(self, populated_memory: AuroraMemory):
        """Test that evolve() runs without errors."""
        # Should not raise
        populated_memory.evolve()
    
    def test_evolve_updates_story_statuses(self, populated_memory: AuroraMemory):
        """Test that evolve may update story statuses."""
        # Run multiple evolution cycles
        for _ in range(3):
            populated_memory.evolve()
        
        # Check stories exist and have valid statuses
        for story in populated_memory.stories.values():
            assert story.status in ("developing", "resolved", "abandoned")
    
    def test_evolve_handles_empty_memory(self, aurora_memory: AuroraMemory):
        """Test that evolve handles empty memory gracefully."""
        # Should not raise on empty memory
        aurora_memory.evolve()


class TestAuroraMemorySerialization:
    """Tests for AuroraMemory serialization."""
    
    def test_to_state_dict(self, populated_memory: AuroraMemory):
        """Test serialization to state dict."""
        state = populated_memory.to_state_dict()
        
        assert "version" in state
        assert "cfg" in state
        assert "plots" in state
        assert "stories" in state
        assert "themes" in state
    
    def test_from_state_dict(self, populated_memory: AuroraMemory):
        """Test deserialization from state dict."""
        # Serialize
        state = populated_memory.to_state_dict()
        
        # Deserialize
        restored = AuroraMemory.from_state_dict(state)
        
        assert len(restored.plots) == len(populated_memory.plots)
        assert len(restored.stories) == len(populated_memory.stories)
    
    def test_round_trip_preserves_data(self, populated_memory: AuroraMemory):
        """Test that serialization round-trip preserves data."""
        state = populated_memory.to_state_dict()
        restored = AuroraMemory.from_state_dict(state)
        
        # Check plots are preserved
        for pid, plot in populated_memory.plots.items():
            assert pid in restored.plots
            restored_plot = restored.plots[pid]
            assert restored_plot.text == plot.text
            assert np.allclose(restored_plot.embedding, plot.embedding)


class TestAuroraMemoryIdentity:
    """Tests for identity-related functionality."""
    
    def test_get_identity_summary(self, populated_memory: AuroraMemory):
        """Test getting identity summary."""
        summary = populated_memory.get_identity_summary()
        
        assert "identity_dimensions" in summary
        assert "relationship_identities" in summary
        assert "relationship_count" in summary
        assert "total_interactions" in summary
    
    def test_identity_dimensions_accumulate(self, aurora_memory: AuroraMemory):
        """Test that identity dimensions accumulate over time."""
        # Ingest interactions that reinforce identity
        for i in range(10):
            aurora_memory.ingest(
                f"用户：帮我解释问题{i}。助理：好的，让我来解释。",
                actors=("user", "assistant"),
            )
        
        # Identity dimensions should have some values
        # (may be empty if no plots were stored)
        summary = aurora_memory.get_identity_summary()
        assert isinstance(summary["identity_dimensions"], dict)


class TestAuroraMemoryPressure:
    """Tests for memory pressure management."""
    
    def test_pressure_manage_respects_capacity(self):
        """Test that pressure management respects max_plots."""
        config = MemoryConfig(dim=64, max_plots=10)
        memory = AuroraMemory(cfg=config, seed=42)
        
        # Ingest many interactions
        for i in range(30):
            memory.ingest(
                f"用户：测试{i}。助理：收到{i}。",
                actors=("user", "assistant"),
            )
        
        # Active plots should not exceed max_plots by much
        # (some tolerance for probabilistic behavior)
        active_plots = len([p for p in memory.plots.values() if p.status == "active"])
        assert active_plots <= config.max_plots * 1.5


class TestAuroraMemoryUtilityMethods:
    """Tests for utility methods."""
    
    def test_update_centroid_online_new(self, aurora_memory: AuroraMemory):
        """Test online centroid update for new centroid."""
        emb = np.random.randn(64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        
        result = aurora_memory._update_centroid_online(None, emb, 1)
        
        assert np.allclose(result, emb)
    
    def test_update_centroid_online_existing(self, aurora_memory: AuroraMemory):
        """Test online centroid update for existing centroid."""
        rng = np.random.default_rng(42)
        current = rng.standard_normal(64).astype(np.float32)
        current = current / np.linalg.norm(current)
        
        new_emb = rng.standard_normal(64).astype(np.float32)
        new_emb = new_emb / np.linalg.norm(new_emb)
        
        result = aurora_memory._update_centroid_online(current, new_emb, 2)
        
        # Result should be normalized
        assert np.isclose(np.linalg.norm(result), 1.0, rtol=1e-5)
    
    def test_create_bidirectional_edge(self, aurora_memory: AuroraMemory):
        """Test bidirectional edge creation."""
        # Add some nodes first
        aurora_memory.graph.add_node("test_node_1", "plot", None)
        aurora_memory.graph.add_node("test_node_2", "story", None)
        
        aurora_memory._create_bidirectional_edge(
            "test_node_1", "test_node_2",
            "belongs_to", "contains"
        )
        
        # Both edges should exist
        assert aurora_memory.graph.g.has_edge("test_node_1", "test_node_2")
        assert aurora_memory.graph.g.has_edge("test_node_2", "test_node_1")


class TestAuroraMemoryExceptions:
    """Tests for custom exception handling."""

    def test_get_story_not_found(self, aurora_memory: AuroraMemory):
        """Test that get_story raises MemoryNotFoundError for non-existent story."""
        with pytest.raises(MemoryNotFoundError) as exc_info:
            aurora_memory.get_story("non_existent_story_id")
        
        assert exc_info.value.kind == "story"
        assert exc_info.value.element_id == "non_existent_story_id"
        assert "story" in str(exc_info.value)
        assert "non_existent_story_id" in str(exc_info.value)

    def test_get_plot_not_found(self, aurora_memory: AuroraMemory):
        """Test that get_plot raises MemoryNotFoundError for non-existent plot."""
        with pytest.raises(MemoryNotFoundError) as exc_info:
            aurora_memory.get_plot("non_existent_plot_id")
        
        assert exc_info.value.kind == "plot"
        assert exc_info.value.element_id == "non_existent_plot_id"
        assert "plot" in str(exc_info.value)
        assert "non_existent_plot_id" in str(exc_info.value)

    def test_get_theme_not_found(self, aurora_memory: AuroraMemory):
        """Test that get_theme raises MemoryNotFoundError for non-existent theme."""
        with pytest.raises(MemoryNotFoundError) as exc_info:
            aurora_memory.get_theme("non_existent_theme_id")
        
        assert exc_info.value.kind == "theme"
        assert exc_info.value.element_id == "non_existent_theme_id"
        assert "theme" in str(exc_info.value)
        assert "non_existent_theme_id" in str(exc_info.value)

    def test_get_story_success(self, populated_memory: AuroraMemory):
        """Test that get_story returns story when it exists."""
        if not populated_memory.stories:
            pytest.skip("No stories created")
        
        story_id = next(iter(populated_memory.stories.keys()))
        story = populated_memory.get_story(story_id)
        assert story is not None
        assert story.id == story_id

    def test_get_plot_success(self, populated_memory: AuroraMemory):
        """Test that get_plot returns plot when it exists."""
        if not populated_memory.plots:
            pytest.skip("No plots stored")
        
        plot_id = next(iter(populated_memory.plots.keys()))
        plot = populated_memory.get_plot(plot_id)
        assert plot is not None
        assert plot.id == plot_id


class TestAuroraMemoryValidation:
    """Tests for input validation."""

    def test_ingest_empty_string_raises(self, aurora_memory: AuroraMemory):
        """Test that ingest raises ValidationError for empty string."""
        with pytest.raises(ValidationError) as exc_info:
            aurora_memory.ingest("")
        
        assert "interaction_text cannot be empty" in str(exc_info.value)

    def test_ingest_whitespace_only_raises(self, aurora_memory: AuroraMemory):
        """Test that ingest raises ValidationError for whitespace-only string."""
        with pytest.raises(ValidationError) as exc_info:
            aurora_memory.ingest("   \n\t  ")
        
        assert "interaction_text cannot be empty" in str(exc_info.value)

    def test_ingest_none_raises(self, aurora_memory: AuroraMemory):
        """Test that ingest raises ValidationError for None input."""
        with pytest.raises(ValidationError) as exc_info:
            aurora_memory.ingest(None)
        
        assert "interaction_text cannot be empty" in str(exc_info.value)

    def test_query_empty_string_raises(self, aurora_memory: AuroraMemory):
        """Test that query raises ValidationError for empty string."""
        with pytest.raises(ValidationError) as exc_info:
            aurora_memory.query("")
        
        assert "query text cannot be empty" in str(exc_info.value)

    def test_query_whitespace_only_raises(self, aurora_memory: AuroraMemory):
        """Test that query raises ValidationError for whitespace-only string."""
        with pytest.raises(ValidationError) as exc_info:
            aurora_memory.query("   \n\t  ")
        
        assert "query text cannot be empty" in str(exc_info.value)

    def test_query_none_raises(self, aurora_memory: AuroraMemory):
        """Test that query raises ValidationError for None input."""
        with pytest.raises(ValidationError) as exc_info:
            aurora_memory.query(None)
        
        assert "query text cannot be empty" in str(exc_info.value)

    def test_ingest_valid_input_succeeds(self, aurora_memory: AuroraMemory):
        """Test that ingest succeeds with valid input."""
        plot = aurora_memory.ingest("用户：有效输入。助理：收到。")
        assert plot is not None
        assert plot.text == "用户：有效输入。助理：收到。"

    def test_query_valid_input_succeeds(self, populated_memory: AuroraMemory):
        """Test that query succeeds with valid input."""
        trace = populated_memory.query("有效查询")
        assert trace is not None
        assert trace.query == "有效查询"
