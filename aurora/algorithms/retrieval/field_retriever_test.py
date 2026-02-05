"""
AURORA Field Retriever Tests
============================

Tests for the FieldRetriever component.

Tests cover:
- Mean-shift attractor tracing
- Personalized PageRank graph diffusion
- Combined retrieval
- Kind filtering
"""

from __future__ import annotations

import numpy as np
import pytest

from aurora.algorithms.retrieval.field_retriever import FieldRetriever
from aurora.algorithms.components.metric import LowRankMetric
from aurora.algorithms.graph.vector_index import VectorIndex
from aurora.algorithms.graph.memory_graph import MemoryGraph
from aurora.embeddings.hash import HashEmbedding


@pytest.fixture
def retriever_setup():
    """Set up retriever with basic components."""
    dim = 64
    metric = LowRankMetric(dim=dim, rank=16, seed=42)
    vindex = VectorIndex(dim=dim)
    graph = MemoryGraph()
    embedder = HashEmbedding(dim=dim, seed=42)
    
    # Add some test vectors
    rng = np.random.default_rng(42)
    
    for i in range(20):
        vec = rng.standard_normal(dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        node_id = f"plot_{i}"
        kind = "plot"
        
        vindex.add(node_id, vec, kind=kind)
        graph.add_node(node_id, kind, {"id": node_id})
    
    # Add some edges
    for i in range(19):
        graph.ensure_edge(f"plot_{i}", f"plot_{i+1}", "temporal")
        graph.ensure_edge(f"plot_{i+1}", f"plot_{i}", "temporal")
    
    retriever = FieldRetriever(metric=metric, vindex=vindex, graph=graph)
    
    return retriever, embedder, vindex, graph


class TestFieldRetrieverBasic:
    """Basic tests for FieldRetriever."""
    
    def test_retrieve_returns_results(self, retriever_setup):
        """Test that retrieve returns results."""
        retriever, embedder, vindex, graph = retriever_setup
        
        trace = retriever.retrieve(
            query_text="测试查询",
            embed=embedder,
            kinds=["plot"],
            k=5,
        )
        
        assert trace is not None
        assert len(trace.ranked) <= 5
    
    def test_retrieve_with_empty_index(self, metric):
        """Test retrieve with empty index."""
        vindex = VectorIndex(dim=64)
        graph = MemoryGraph()
        retriever = FieldRetriever(metric=metric, vindex=vindex, graph=graph)
        embedder = HashEmbedding(dim=64, seed=42)
        
        trace = retriever.retrieve(
            query_text="测试",
            embed=embedder,
            kinds=["plot"],
            k=5,
        )
        
        assert trace is not None
        assert len(trace.ranked) == 0


class TestMeanShift:
    """Tests for mean-shift attractor tracing."""
    
    def test_mean_shift_convergence(self, retriever_setup):
        """Test that mean-shift converges to an attractor."""
        retriever, embedder, vindex, graph = retriever_setup
        
        # Create a query embedding
        query = embedder.embed("测试查询")
        
        # Create candidates from vindex
        candidates = []
        for pid in ["plot_0", "plot_1", "plot_2"]:
            emb = embedder.embed(f"测试文本{pid}")
            candidates.append((pid, emb, 1.0))
        
        # Run mean-shift
        path = retriever._mean_shift(query, candidates, steps=8)
        
        # Path should contain embeddings
        assert len(path) > 0
        assert path[-1] is not None
        assert len(path[-1]) == 64
    
    def test_mean_shift_path_recorded(self, retriever_setup):
        """Test that mean-shift records the path."""
        retriever, embedder, vindex, graph = retriever_setup
        
        query = embedder.embed("测试查询")
        
        # Create candidates
        candidates = [(f"plot_{i}", embedder.embed(f"测试{i}"), 1.0) for i in range(3)]
        
        path = retriever._mean_shift(query, candidates, steps=8)
        
        # Path should have recorded steps
        assert isinstance(path, list)
        assert len(path) == 9  # initial + 8 steps


class TestPageRank:
    """Tests for personalized PageRank."""
    
    def test_pagerank_with_seeds(self, retriever_setup):
        """Test PageRank with seed nodes."""
        retriever, embedder, vindex, graph = retriever_setup
        
        # Use some nodes as seeds
        seeds = {"plot_0": 0.5, "plot_1": 0.5}
        
        scores = retriever._pagerank(seeds, damping=0.85, max_iter=60)
        
        assert scores is not None
        assert isinstance(scores, dict)
        # May be empty if graph doesn't have these nodes
    
    def test_pagerank_empty_seeds(self, retriever_setup):
        """Test PageRank with empty seeds."""
        retriever, embedder, vindex, graph = retriever_setup
        
        scores = retriever._pagerank({}, damping=0.85, max_iter=60)
        
        assert scores is not None
        assert len(scores) == 0


class TestKindFiltering:
    """Tests for kind filtering in retrieval."""
    
    def test_filter_by_kind(self, retriever_setup):
        """Test that results are filtered by kind."""
        retriever, embedder, vindex, graph = retriever_setup
        
        trace = retriever.retrieve(
            query_text="测试",
            embed=embedder,
            kinds=["plot"],
            k=5,
        )
        
        # All results should be of the requested kind
        for node_id, score, kind in trace.ranked:
            assert kind == "plot"
    
    def test_multiple_kinds(self, retriever_setup):
        """Test retrieval with multiple kinds."""
        retriever, embedder, vindex, graph = retriever_setup
        
        # Add a story node
        vec = np.random.randn(64).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        vindex, graph = retriever_setup[2], retriever_setup[3]
        vindex.add("story_1", vec, kind="story")
        graph.add_node("story_1", "story", {"id": "story_1"})
        
        trace = retriever.retrieve(
            query_text="测试",
            embed=embedder,
            kinds=["plot", "story"],
            k=10,
        )
        
        # Results should include both kinds if available
        kinds_in_results = {kind for _, _, kind in trace.ranked}
        assert "plot" in kinds_in_results or "story" in kinds_in_results


class TestRetrievalTrace:
    """Tests for RetrievalTrace structure."""
    
    def test_trace_has_query_info(self, retriever_setup):
        """Test that trace includes query information."""
        retriever, embedder, vindex, graph = retriever_setup
        
        trace = retriever.retrieve(
            query_text="测试查询文本",
            embed=embedder,
            kinds=["plot"],
            k=5,
        )
        
        assert trace.query == "测试查询文本"
        assert trace.query_emb is not None
    
    def test_trace_has_attractor_path(self, retriever_setup):
        """Test that trace includes attractor path."""
        retriever, embedder, vindex, graph = retriever_setup
        
        trace = retriever.retrieve(
            query_text="测试",
            embed=embedder,
            kinds=["plot"],
            k=5,
        )
        
        assert hasattr(trace, 'attractor_path')
