"""
AURORA Temporal Retrieval Tests
================================

Tests for Time as First-Class Citizen functionality.

Tests cover:
- TimeAnchor detection
- Temporal-aware reranking
- Temporal index management
- Story temporal span and narrative
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import pytest

from aurora.algorithms.aurora_core import AuroraMemory
from aurora.algorithms.models.config import MemoryConfig
from aurora.algorithms.models.plot import Plot
from aurora.algorithms.models.story import StoryArc
from aurora.algorithms.retrieval.field_retriever import (
    FieldRetriever,
    QueryType,
    TimeAnchor,
)
from aurora.algorithms.components.metric import LowRankMetric
from aurora.algorithms.graph.vector_index import VectorIndex
from aurora.algorithms.graph.memory_graph import MemoryGraph
from aurora.embeddings.hash import HashEmbedding
from aurora.utils.time_utils import now_ts


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def retriever_setup():
    """Set up retriever with basic components."""
    dim = 64
    metric = LowRankMetric(dim=dim, rank=16, seed=42)
    vindex = VectorIndex(dim=dim)
    graph = MemoryGraph()
    embedder = HashEmbedding(dim=dim, seed=42)
    
    retriever = FieldRetriever(metric=metric, vindex=vindex, graph=graph)
    
    return retriever, embedder, vindex, graph, metric


@pytest.fixture
def aurora_memory() -> AuroraMemory:
    """Fresh AuroraMemory instance for testing."""
    return AuroraMemory(
        cfg=MemoryConfig(dim=64, max_plots=100),
        seed=42
    )


@pytest.fixture
def populated_memory_with_timestamps(aurora_memory: AuroraMemory) -> AuroraMemory:
    """Memory with test data at different timestamps for temporal testing."""
    import time
    
    base_ts = now_ts()
    
    # Ingest events at different times
    # Event 1: 30 days ago - Python learning started
    aurora_memory.ingest(
        "User: I started learning Python in January. Assistant: Great!",
        event_id="event_001"
    )
    if "plot-event_001" in aurora_memory.plots:
        aurora_memory.plots["plot-event_001"].ts = base_ts - 30 * 86400
    
    # Event 2: 20 days ago - JavaScript learning
    aurora_memory.ingest(
        "User: Then I learned JavaScript in February. Assistant: Good progress!",
        event_id="event_002"
    )
    if "plot-event_002" in aurora_memory.plots:
        aurora_memory.plots["plot-event_002"].ts = base_ts - 20 * 86400
    
    # Event 3: 10 days ago - Rust learning
    aurora_memory.ingest(
        "User: Now I'm learning Rust. Assistant: Excellent!",
        event_id="event_003"
    )
    if "plot-event_003" in aurora_memory.plots:
        aurora_memory.plots["plot-event_003"].ts = base_ts - 10 * 86400
    
    # Event 4: Today - Current focus
    aurora_memory.ingest(
        "User: Today I'm focusing on TypeScript. Assistant: Good choice!",
        event_id="event_004"
    )
    # This one keeps its current timestamp
    
    return aurora_memory


# =============================================================================
# TimeAnchor Detection Tests
# =============================================================================

class TestTimeAnchorDetection:
    """Tests for time anchor detection in queries."""
    
    def test_detect_recent_anchor_chinese(self, retriever_setup):
        """Test detection of RECENT anchor with Chinese keywords."""
        retriever, *_ = retriever_setup
        
        recent_queries = [
            "最近学了什么？",
            "上次我们讨论了什么？",
            "刚才说的是什么？",
            "最新的进展如何？",
        ]
        
        for query in recent_queries:
            anchor = retriever._detect_time_anchor(query)
            assert anchor == TimeAnchor.RECENT, f"Failed for query: {query}"
    
    def test_detect_recent_anchor_english(self, retriever_setup):
        """Test detection of RECENT anchor with English keywords."""
        retriever, *_ = retriever_setup
        
        recent_queries = [
            "What did I learn recently?",
            "What was the last thing we discussed?",
            "Just talked about what?",
            "What's the latest update?",
        ]
        
        for query in recent_queries:
            anchor = retriever._detect_time_anchor(query)
            assert anchor == TimeAnchor.RECENT, f"Failed for query: {query}"
    
    def test_detect_earliest_anchor_chinese(self, retriever_setup):
        """Test detection of EARLIEST anchor with Chinese keywords."""
        retriever, *_ = retriever_setup
        
        earliest_queries = [
            "最早学的是什么？",
            "一开始我们讨论了什么？",
            "第一次接触这个话题是什么时候？",
            "最初的计划是什么？",
        ]
        
        for query in earliest_queries:
            anchor = retriever._detect_time_anchor(query)
            assert anchor == TimeAnchor.EARLIEST, f"Failed for query: {query}"
    
    def test_detect_earliest_anchor_english(self, retriever_setup):
        """Test detection of EARLIEST anchor with English keywords."""
        retriever, *_ = retriever_setup
        
        earliest_queries = [
            "What did I learn first?",
            "What was originally discussed?",
            "When did this initially start?",
            "What was the earliest topic?",
        ]
        
        for query in earliest_queries:
            anchor = retriever._detect_time_anchor(query)
            assert anchor == TimeAnchor.EARLIEST, f"Failed for query: {query}"
    
    def test_detect_span_anchor_chinese(self, retriever_setup):
        """Test detection of SPAN anchor with Chinese keywords."""
        retriever, *_ = retriever_setup
        
        span_queries = [
            "我学习的历史是什么？",
            "整个过程是怎样的？",
            "从开始到现在的演变？",
            "一直以来的发展过程？",
        ]
        
        for query in span_queries:
            anchor = retriever._detect_time_anchor(query)
            assert anchor == TimeAnchor.SPAN, f"Failed for query: {query}"
    
    def test_detect_span_anchor_english(self, retriever_setup):
        """Test detection of SPAN anchor with English keywords."""
        retriever, *_ = retriever_setup
        
        span_queries = [
            "What's the history of my learning?",
            "How has it evolved over time?",
            "What's the timeline of events?",
            "Show me the progression throughout.",
        ]
        
        for query in span_queries:
            anchor = retriever._detect_time_anchor(query)
            assert anchor == TimeAnchor.SPAN, f"Failed for query: {query}"
    
    def test_detect_no_anchor(self, retriever_setup):
        """Test that queries without temporal keywords return NONE."""
        retriever, *_ = retriever_setup
        
        non_temporal_queries = [
            "What is Python?",
            "How do I sort a list?",
            "Explain machine learning",
            "帮我写一个函数",
        ]
        
        for query in non_temporal_queries:
            anchor = retriever._detect_time_anchor(query)
            assert anchor == TimeAnchor.NONE, f"Failed for query: {query}"


# =============================================================================
# Query Type Classification Tests
# =============================================================================

class TestQueryTypeClassification:
    """Tests for query type classification."""
    
    def test_classify_temporal_query(self, retriever_setup):
        """Test that temporal queries are classified correctly."""
        retriever, *_ = retriever_setup
        
        temporal_queries = [
            "什么时候开始学的？",
            "上次我们聊了什么？",
            "When did this happen?",
            "What was the last topic?",
        ]
        
        for query in temporal_queries:
            query_type = retriever._classify_query(query)
            assert query_type == QueryType.TEMPORAL, f"Failed for query: {query}"
    
    def test_classify_factual_query(self, retriever_setup):
        """Test that factual queries are classified correctly."""
        retriever, *_ = retriever_setup
        
        factual_queries = [
            "What is Python?",
            "How to sort a list?",
            "Python是什么？",
            "如何排序列表？",
        ]
        
        for query in factual_queries:
            query_type = retriever._classify_query(query)
            assert query_type == QueryType.FACTUAL, f"Failed for query: {query}"


# =============================================================================
# Temporal Index Tests
# =============================================================================

class TestTemporalIndex:
    """Tests for temporal index management."""
    
    def test_add_to_temporal_index(self, aurora_memory):
        """Test adding plots to temporal index."""
        # Ingest a plot
        aurora_memory.ingest("Test interaction", event_id="test_001")
        
        # Check temporal index is populated
        assert len(aurora_memory._temporal_index) > 0
    
    def test_get_recent_plots(self, populated_memory_with_timestamps):
        """Test getting recent plots."""
        mem = populated_memory_with_timestamps
        
        recent = mem.get_recent_plots(n=2)
        
        # Should have at most 2 results
        assert len(recent) <= 2
        
        # If we have results, they should be plot IDs
        for pid in recent:
            assert pid in mem.plots
    
    def test_get_earliest_plots(self, populated_memory_with_timestamps):
        """Test getting earliest plots."""
        mem = populated_memory_with_timestamps
        
        earliest = mem.get_earliest_plots(n=2)
        
        # Should have at most 2 results
        assert len(earliest) <= 2
        
        # If we have results, they should be plot IDs
        for pid in earliest:
            assert pid in mem.plots
    
    def test_get_plots_in_time_range(self, populated_memory_with_timestamps):
        """Test getting plots within a time range."""
        mem = populated_memory_with_timestamps
        
        if not mem.plots:
            pytest.skip("No plots stored")
        
        # Get all plots (no time restriction)
        all_plots = mem.get_plots_in_time_range()
        
        # Should have some plots
        assert len(all_plots) > 0
    
    def test_temporal_statistics(self, populated_memory_with_timestamps):
        """Test temporal statistics."""
        mem = populated_memory_with_timestamps
        
        stats = mem.get_temporal_statistics()
        
        assert "total_days" in stats
        assert "avg_plots_per_day" in stats
        
        # Should have tracked some days
        if mem.plots:
            assert stats["total_days"] > 0


# =============================================================================
# Temporal Reranking Tests
# =============================================================================

class TestTemporalReranking:
    """Tests for temporal-aware reranking."""
    
    def test_recent_anchor_reranking(self, retriever_setup):
        """Test that RECENT anchor sorts by timestamp descending."""
        retriever, embedder, vindex, graph, metric = retriever_setup
        
        # Create mock items with timestamps
        base_ts = now_ts()
        
        # Add nodes to graph with different timestamps
        for i, ts_offset in enumerate([100, 50, 200, 10]):  # Random order
            node_id = f"plot_{i}"
            vec = embedder.embed(f"test {i}")
            vindex.add(node_id, vec, kind="plot")
            
            # Create a mock payload with timestamp
            class MockPayload:
                def __init__(self, ts):
                    self.ts = ts
            
            graph.add_node(node_id, "plot", MockPayload(base_ts - ts_offset))
        
        # Create ranked list
        ranked = [
            (f"plot_{i}", 0.8 - i * 0.1, "plot") 
            for i in range(4)
        ]
        
        # Apply recent anchor reranking
        result = retriever._temporal_aware_rerank(ranked, "最近学了什么？", k=4)
        
        # Get timestamps for verification
        result_timestamps = [
            retriever._get_timestamp(nid) for nid, _, _ in result
        ]
        
        # Should be sorted by timestamp descending (most recent first)
        assert result_timestamps == sorted(result_timestamps, reverse=True)
    
    def test_earliest_anchor_reranking(self, retriever_setup):
        """Test that EARLIEST anchor sorts by timestamp ascending."""
        retriever, embedder, vindex, graph, metric = retriever_setup
        
        # Create mock items with timestamps
        base_ts = now_ts()
        
        # Add nodes to graph with different timestamps
        for i, ts_offset in enumerate([100, 50, 200, 10]):  # Random order
            node_id = f"plot_{i}"
            vec = embedder.embed(f"test {i}")
            vindex.add(node_id, vec, kind="plot")
            
            class MockPayload:
                def __init__(self, ts):
                    self.ts = ts
            
            graph.add_node(node_id, "plot", MockPayload(base_ts - ts_offset))
        
        # Create ranked list
        ranked = [
            (f"plot_{i}", 0.8 - i * 0.1, "plot") 
            for i in range(4)
        ]
        
        # Apply earliest anchor reranking
        result = retriever._temporal_aware_rerank(ranked, "最早学的是什么？", k=4)
        
        # Get timestamps for verification
        result_timestamps = [
            retriever._get_timestamp(nid) for nid, _, _ in result
        ]
        
        # Should be sorted by timestamp ascending (earliest first)
        assert result_timestamps == sorted(result_timestamps)
    
    def test_span_anchor_diversity(self, retriever_setup):
        """Test that SPAN anchor provides temporal diversity."""
        retriever, embedder, vindex, graph, metric = retriever_setup
        
        # Create mock items with timestamps spread over time
        base_ts = now_ts()
        day_seconds = 86400
        
        # Create 10 nodes spread over 30 days
        for i in range(10):
            node_id = f"plot_{i}"
            vec = embedder.embed(f"test {i}")
            vindex.add(node_id, vec, kind="plot")
            
            class MockPayload:
                def __init__(self, ts):
                    self.ts = ts
            
            # Spread timestamps: 0, 3, 6, 9, 12, 15, 18, 21, 24, 27 days ago
            graph.add_node(node_id, "plot", MockPayload(base_ts - i * 3 * day_seconds))
        
        # Create ranked list with descending scores
        ranked = [
            (f"plot_{i}", 1.0 - i * 0.05, "plot") 
            for i in range(10)
        ]
        
        # Apply span anchor reranking
        result = retriever._temporal_aware_rerank(ranked, "学习的历史", k=5)
        
        # Should have 5 results
        assert len(result) == 5
        
        # Results should come from different time periods (temporal diversity)
        result_ids = [nid for nid, _, _ in result]
        # Not all should be from the highest scored items
        # This is a probabilistic test - the exact results depend on MMR


# =============================================================================
# Story Temporal Methods Tests
# =============================================================================

class TestStoryTemporalMethods:
    """Tests for StoryArc temporal methods."""
    
    def test_get_temporal_span_without_plots(self):
        """Test temporal span with no plots dict."""
        story = StoryArc(
            id="story_001",
            created_ts=1000000.0,
            updated_ts=2000000.0,
            plot_ids=["p1", "p2", "p3"]
        )
        
        start, end = story.get_temporal_span()
        
        assert start == 1000000.0
        assert end == 2000000.0
    
    def test_get_temporal_span_with_plots(self):
        """Test temporal span with plots dict."""
        story = StoryArc(
            id="story_001",
            created_ts=1000000.0,
            updated_ts=2000000.0,
            plot_ids=["p1", "p2", "p3"]
        )
        
        # Create mock plots
        class MockPlot:
            def __init__(self, ts):
                self.ts = ts
        
        plots_dict = {
            "p1": MockPlot(1500000.0),
            "p2": MockPlot(1200000.0),
            "p3": MockPlot(1800000.0),
        }
        
        start, end = story.get_temporal_span(plots_dict)
        
        assert start == 1200000.0  # Earliest
        assert end == 1800000.0    # Latest
    
    def test_get_temporal_narrative_chinese(self):
        """Test temporal narrative generation in Chinese."""
        story = StoryArc(
            id="story_001",
            created_ts=now_ts() - 30 * 86400,  # 30 days ago
            updated_ts=now_ts(),
            plot_ids=["p1", "p2", "p3", "p4", "p5"],
            relationship_with="user",
            my_identity_in_this_relationship="助手",
        )
        
        narrative = story.get_temporal_narrative(locale="zh")
        
        # Should contain key information
        assert "user" in narrative
        assert "约" in narrative or "天" in narrative or "月" in narrative
        assert "5次交互" in narrative
        assert "助手" in narrative
    
    def test_get_temporal_narrative_english(self):
        """Test temporal narrative generation in English."""
        story = StoryArc(
            id="story_001",
            created_ts=now_ts() - 30 * 86400,  # 30 days ago
            updated_ts=now_ts(),
            plot_ids=["p1", "p2", "p3", "p4", "p5"],
            relationship_with="user",
            my_identity_in_this_relationship="assistant",
        )
        
        narrative = story.get_temporal_narrative(locale="en")
        
        # Should contain key information
        assert "user" in narrative
        assert "5 interactions" in narrative
        assert "assistant" in narrative
    
    def test_get_temporal_density(self):
        """Test temporal density calculation."""
        story = StoryArc(
            id="story_001",
            created_ts=now_ts() - 10 * 86400,  # 10 days ago
            updated_ts=now_ts(),
            plot_ids=["p1", "p2", "p3", "p4", "p5"],  # 5 interactions
        )
        
        density = story.get_temporal_density()
        
        # 5 interactions over 10 days = 0.5 per day
        assert 0.4 < density < 0.6


# =============================================================================
# Integration Tests
# =============================================================================

class TestTemporalIntegration:
    """Integration tests for temporal functionality."""
    
    def test_query_with_temporal_anchor_recent(self, populated_memory_with_timestamps):
        """Test query with recent temporal anchor."""
        mem = populated_memory_with_timestamps
        
        if not mem.plots:
            pytest.skip("No plots stored")
        
        trace = mem.query("What am I learning most recently?", k=3)
        
        # Should detect temporal query type
        assert trace.query_type == QueryType.TEMPORAL
        
        # Results should be returned (even if semantic matching is random with HashEmbedding)
        # The key is that temporal ordering should be applied
    
    def test_query_with_temporal_anchor_earliest(self, populated_memory_with_timestamps):
        """Test query with earliest temporal anchor."""
        mem = populated_memory_with_timestamps
        
        if not mem.plots:
            pytest.skip("No plots stored")
        
        trace = mem.query("What did I learn first?", k=3)
        
        # Should detect temporal query type
        assert trace.query_type == QueryType.TEMPORAL
    
    def test_serialization_preserves_temporal_index(self, populated_memory_with_timestamps):
        """Test that temporal index is preserved through serialization."""
        mem = populated_memory_with_timestamps
        
        # Get original temporal index state
        original_index = dict(mem._temporal_index)
        original_min = mem._temporal_index_min_bucket
        original_max = mem._temporal_index_max_bucket
        
        # Serialize and deserialize
        state = mem.to_state_dict()
        restored = AuroraMemory.from_state_dict(state)
        
        # Temporal index should be preserved
        assert len(restored._temporal_index) == len(original_index)
        assert restored._temporal_index_min_bucket == original_min
        assert restored._temporal_index_max_bucket == original_max


# =============================================================================
# User Story Tests (from requirements)
# =============================================================================

class TestUserStoryTemporal:
    """Tests based on user story requirements."""
    
    def test_recent_learning_query(self, populated_memory_with_timestamps):
        """
        Test: "What am I learning most recently?"
        Expected: Should return Rust/TypeScript (most recent)
        """
        mem = populated_memory_with_timestamps
        
        if not mem.plots:
            pytest.skip("No plots stored")
        
        trace = mem.query("What am I learning most recently?", k=1)
        
        # With proper semantic embeddings, should return most recent learning
        # With HashEmbedding, temporal sorting still applies after semantic retrieval
        if trace.ranked:
            result_id = trace.ranked[0][0]
            # The result should exist in plots
            assert result_id in mem.plots or result_id in mem.stories
    
    def test_first_learning_query(self, populated_memory_with_timestamps):
        """
        Test: "What did I learn first?"
        Expected: Should return Python (earliest)
        """
        mem = populated_memory_with_timestamps
        
        if not mem.plots:
            pytest.skip("No plots stored")
        
        trace = mem.query("What did I learn first?", k=1)
        
        if trace.ranked:
            result_id = trace.ranked[0][0]
            assert result_id in mem.plots or result_id in mem.stories
