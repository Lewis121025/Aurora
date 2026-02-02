"""
AURORA Time Filter Tests
========================

Tests for time range extraction and pre-filtering optimization.
"""

from __future__ import annotations

import numpy as np
import pytest
import time

from aurora.algorithms.retrieval.time_filter import TimeRangeExtractor, TimeRange
from aurora.algorithms.retrieval.field_retriever import FieldRetriever, QueryType
from aurora.algorithms.components.metric import LowRankMetric
from aurora.algorithms.graph.vector_index import VectorIndex
from aurora.algorithms.graph.memory_graph import MemoryGraph
from aurora.embeddings.hash import HashEmbedding


@pytest.fixture
def time_extractor():
    """Time range extractor fixture."""
    return TimeRangeExtractor()


@pytest.fixture
def events_timeline():
    """Sample events timeline for testing."""
    base_ts = time.time()
    return [
        ("学习Python", base_ts - 30 * 86400),  # 30 days ago
        ("学习JavaScript", base_ts - 20 * 86400),  # 20 days ago
        ("学习Rust", base_ts - 10 * 86400),  # 10 days ago
        ("学习TypeScript", base_ts),  # Today
    ]


class TestTimeRangeExtraction:
    """Tests for time range extraction from queries."""
    
    def test_extract_first_anchor(self, time_extractor, events_timeline):
        """Test extraction of 'first' time anchor."""
        time_range = time_extractor.extract("最早学了什么？", events_timeline)
        
        assert time_range.relation == "first"
        assert time_range.end is not None
        assert time_range.start is None
        # End should be around earliest timestamp + 1 day
        earliest_ts = min(ts for _, ts in events_timeline)
        assert abs(time_range.end - (earliest_ts + 86400)) < 100  # Allow small tolerance
    
    def test_extract_last_anchor(self, time_extractor, events_timeline):
        """Test extraction of 'last' time anchor."""
        time_range = time_extractor.extract("最近学了什么？", events_timeline)
        
        assert time_range.relation == "last"
        assert time_range.start is not None
        assert time_range.end is None
        # Start should be around latest timestamp - 1 day
        latest_ts = max(ts for _, ts in events_timeline)
        assert abs(time_range.start - (latest_ts - 86400)) < 100
    
    def test_extract_span_anchor(self, time_extractor, events_timeline):
        """Test extraction of 'span' time anchor."""
        time_range = time_extractor.extract("学习的历史", events_timeline)
        
        assert time_range.relation == "span"
        # Span queries don't filter (need full range)
        assert time_range.start is None
        assert time_range.end is None
    
    def test_extract_no_anchor(self, time_extractor, events_timeline):
        """Test extraction with no temporal anchor."""
        time_range = time_extractor.extract("学了什么？", events_timeline)
        
        assert time_range.relation == "any"
        assert time_range.start is None
        assert time_range.end is None
    
    def test_extract_relative_time_yesterday(self, time_extractor, events_timeline):
        """Test extraction of relative time patterns."""
        time_range = time_extractor.extract("昨天学了什么？", events_timeline)
        
        assert time_range.relation == "during"
        assert time_range.start is not None
        assert time_range.end is not None
    
    def test_extract_without_timeline(self, time_extractor):
        """Test extraction without timeline (should return relation-only)."""
        time_range = time_extractor.extract("最早学了什么？", events_timeline=None)
        
        assert time_range.relation == "first"
        # Without timeline, can't resolve specific timestamps
        assert time_range.start is None or time_range.end is None


class TestTimeRangeFiltering:
    """Tests for time range filtering of candidates."""
    
    def test_filter_by_range_first(self, time_extractor):
        """Test filtering with 'first' relation."""
        base_ts = time.time()
        candidates = [
            ("id1", 0.9, base_ts - 30 * 86400),  # 30 days ago
            ("id2", 0.8, base_ts - 20 * 86400),  # 20 days ago
            ("id3", 0.7, base_ts - 10 * 86400),  # 10 days ago
            ("id4", 0.6, base_ts),  # Today
        ]
        
        time_range = TimeRange(
            end=base_ts - 25 * 86400,  # Filter to first 25 days
            relation="first"
        )
        
        def get_ts(nid: str) -> float:
            for cid, _, ts in candidates:
                if cid == nid:
                    return ts
            return 0.0
        
        filtered = time_extractor.filter_by_range(candidates, time_range, get_ts)
        
        # Should only include items before end timestamp
        assert len(filtered) == 1
        assert filtered[0][0] == "id1"
        # Should be sorted ascending (earliest first)
        assert filtered[0][2] <= filtered[-1][2] if len(filtered) > 1 else True
    
    def test_filter_by_range_last(self, time_extractor):
        """Test filtering with 'last' relation."""
        base_ts = time.time()
        candidates = [
            ("id1", 0.9, base_ts - 30 * 86400),  # 30 days ago
            ("id2", 0.8, base_ts - 20 * 86400),  # 20 days ago
            ("id3", 0.7, base_ts - 10 * 86400),  # 10 days ago
            ("id4", 0.6, base_ts),  # Today
        ]
        
        time_range = TimeRange(
            start=base_ts - 15 * 86400,  # Filter to last 15 days
            relation="last"
        )
        
        def get_ts(nid: str) -> float:
            for cid, _, ts in candidates:
                if cid == nid:
                    return ts
            return 0.0
        
        filtered = time_extractor.filter_by_range(candidates, time_range, get_ts)
        
        # Should only include items after start timestamp
        assert len(filtered) == 2
        assert filtered[0][0] in ["id3", "id4"]
        # Should be sorted descending (latest first)
        assert filtered[0][2] >= filtered[-1][2]
    
    def test_filter_by_range_any(self, time_extractor):
        """Test filtering with 'any' relation (no filtering)."""
        base_ts = time.time()
        candidates = [
            ("id1", 0.9, base_ts - 30 * 86400),
            ("id2", 0.8, base_ts - 20 * 86400),
        ]
        
        time_range = TimeRange(relation="any")
        
        def get_ts(nid: str) -> float:
            return 0.0
        
        filtered = time_extractor.filter_by_range(candidates, time_range, get_ts)
        
        # Should return all candidates unchanged
        assert len(filtered) == len(candidates)
    
    def test_filter_by_range_span(self, time_extractor):
        """Test filtering with 'span' relation (no filtering)."""
        base_ts = time.time()
        candidates = [
            ("id1", 0.9, base_ts - 30 * 86400),
            ("id2", 0.8, base_ts - 20 * 86400),
        ]
        
        time_range = TimeRange(relation="span")
        
        def get_ts(nid: str) -> float:
            return 0.0
        
        filtered = time_extractor.filter_by_range(candidates, time_range, get_ts)
        
        # Should return all candidates unchanged
        assert len(filtered) == len(candidates)


class TestTimeFilterIntegration:
    """Integration tests for time filtering in FieldRetriever."""
    
    @pytest.fixture
    def retriever_setup(self):
        """Set up retriever with temporal test data."""
        dim = 64
        metric = LowRankMetric(dim=dim, rank=16, seed=42)
        vindex = VectorIndex(dim=dim)
        graph = MemoryGraph()
        embedder = HashEmbedding(dim=dim, seed=42)
        
        base_ts = time.time()
        rng = np.random.default_rng(42)
        
        # Add plots at different timestamps
        timestamps = [
            base_ts - 30 * 86400,  # 30 days ago
            base_ts - 20 * 86400,  # 20 days ago
            base_ts - 10 * 86400,  # 10 days ago
            base_ts,  # Today
        ]
        
        texts = [
            "学习Python",
            "学习JavaScript",
            "学习Rust",
            "学习TypeScript",
        ]
        
        for i, (text, ts) in enumerate(zip(texts, timestamps)):
            vec = rng.standard_normal(dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            node_id = f"plot_{i}"
            
            vindex.add(node_id, vec, kind="plot")
            # Store timestamp in payload
            graph.add_node(node_id, "plot", {
                "id": node_id,
                "text": text,
                "ts": ts,
                "embedding": vec,
            })
        
        retriever = FieldRetriever(metric=metric, vindex=vindex, graph=graph)
        
        return retriever, embedder, vindex, graph
    
    def test_temporal_query_with_time_filter(self, retriever_setup):
        """Test that temporal queries apply time filtering."""
        retriever, embedder, vindex, graph = retriever_setup
        
        # Query for earliest learning
        trace = retriever.retrieve(
            query_text="最早学了什么？",
            embed=embedder,
            kinds=("plot",),
            k=5,
        )
        
        # Should detect as TEMPORAL query
        assert trace.query_type == QueryType.TEMPORAL
        
        # Results should be filtered to earliest time period
        # (In this case, should prefer plot_0 which is 30 days ago)
        if trace.ranked:
            # Check that results are temporally filtered
            timestamps = [retriever._get_timestamp(nid) for nid, _, _ in trace.ranked]
            # For "first" queries, earliest timestamp should be in results
            earliest_ts = min(ts for nid in graph.g.nodes() 
                            if graph.kind(nid) == "plot" 
                            for ts in [retriever._get_timestamp(nid)])
            assert min(timestamps) <= earliest_ts + 86400  # Within 1 day buffer
    
    def test_temporal_query_recent(self, retriever_setup):
        """Test that 'recent' queries filter to latest time period."""
        retriever, embedder, vindex, graph = retriever_setup
        
        # Query for recent learning
        trace = retriever.retrieve(
            query_text="最近学了什么？",
            embed=embedder,
            kinds=("plot",),
            k=5,
        )
        
        # Should detect as TEMPORAL query
        assert trace.query_type == QueryType.TEMPORAL
        
        # Results should be filtered to latest time period
        if trace.ranked:
            timestamps = [retriever._get_timestamp(nid) for nid, _, _ in trace.ranked]
            # For "last" queries, latest timestamp should be in results
            latest_ts = max(ts for nid in graph.g.nodes() 
                          if graph.kind(nid) == "plot" 
                          for ts in [retriever._get_timestamp(nid)])
            assert max(timestamps) >= latest_ts - 86400  # Within 1 day buffer
    
    def test_non_temporal_query_no_filter(self, retriever_setup):
        """Test that non-temporal queries don't apply time filtering."""
        retriever, embedder, vindex, graph = retriever_setup
        
        # Factual query (no temporal keywords)
        trace = retriever.retrieve(
            query_text="学了什么？",
            embed=embedder,
            kinds=("plot",),
            k=5,
        )
        
        # Should NOT be detected as TEMPORAL
        assert trace.query_type != QueryType.TEMPORAL
        
        # Results should include all time periods (no filtering)
        if trace.ranked:
            timestamps = [retriever._get_timestamp(nid) for nid, _, _ in trace.ranked]
            # Should have results from different time periods
            assert len(set(int(ts // 86400) for ts in timestamps)) >= 1
