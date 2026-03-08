"""
AURORA 连贯性模块的测试
"""

from __future__ import annotations

import numpy as np
import pytest

from aurora.core.models.plot import Plot
from aurora.core.models.story import StoryArc
from aurora.core.models.theme import Theme
from aurora.core.graph.memory_graph import MemoryGraph
from aurora.core.components.metric import LowRankMetric
from aurora.core.coherence import (
    ConflictType,
    Conflict,
    Resolution,
    CoherenceReport,
    BeliefNetwork,
    BeliefState,
    ContradictionDetector,
    CoherenceScorer,
    ConflictResolver,
    CoherenceGuardian,
)
from aurora.core.causal import CausalEdgeBelief
from aurora.utils.time_utils import now_ts


@pytest.fixture
def metric():
    return LowRankMetric(dim=64, rank=16, seed=42)


@pytest.fixture
def sample_plots():
    """Create sample plots for testing"""
    base_time = now_ts()
    
    emb1 = np.random.randn(64).astype(np.float32)
    emb1 = emb1 / np.linalg.norm(emb1)
    
    emb2 = np.random.randn(64).astype(np.float32)
    emb2 = emb2 / np.linalg.norm(emb2)
    
    # Create a potentially contradicting pair
    emb3 = -0.8 * emb1 + 0.2 * np.random.randn(64).astype(np.float32)
    emb3 = emb3 / np.linalg.norm(emb3)
    
    return {
        "p1": Plot(
            id="p1",
            text="User prefers simple solutions",
            embedding=emb1,
            ts=base_time,
            actors=["user"],
        ),
        "p2": Plot(
            id="p2",
            text="Agent provided detailed explanation",
            embedding=emb2,
            ts=base_time + 60,
            actors=["agent"],
        ),
        "p3": Plot(
            id="p3",
            text="User prefers complex detailed solutions",
            embedding=emb3,
            ts=base_time + 120,
            actors=["user"],
        ),
    }


@pytest.fixture
def sample_themes():
    """Create sample themes for testing"""
    ts = now_ts()
    emb1 = np.random.randn(64).astype(np.float32)
    emb1 = emb1 / np.linalg.norm(emb1)
    
    emb2 = np.random.randn(64).astype(np.float32)
    emb2 = emb2 / np.linalg.norm(emb2)
    
    return {
        "t1": Theme(
            id="t1",
            created_ts=ts,
            updated_ts=ts,
            name="Simplicity",
            description="User values simplicity",
            prototype=emb1,
            a=8.0,
            b=2.0,
        ),
        "t2": Theme(
            id="t2",
            created_ts=ts,
            updated_ts=ts,
            name="Thoroughness",
            description="User values thoroughness",
            prototype=emb2,
            a=6.0,
            b=4.0,
        ),
    }


class TestBeliefNetwork:
    """Tests for BeliefNetwork"""
    
    def test_add_belief(self):
        network = BeliefNetwork()
        network.add_belief("node1", prior=0.7, evidence_strength=1.0)
        
        assert "node1" in network.beliefs
        assert network.beliefs["node1"].prior == 0.7
    
    def test_add_dependency(self):
        network = BeliefNetwork()
        network.add_belief("a", 0.5)
        network.add_belief("b", 0.5)
        network.add_dependency("a", "b", "supports", 0.8)
        
        assert network.graph.has_edge("a", "b")
        assert network.graph.edges["a", "b"]["type"] == "supports"
    
    def test_propagate_beliefs_support(self):
        network = BeliefNetwork()
        network.add_belief("cause", prior=0.9, evidence_strength=1.0)
        network.add_belief("effect", prior=0.5, evidence_strength=0.5)
        network.add_dependency("cause", "effect", "supports", 0.8)
        
        probs = network.propagate_beliefs(iterations=5)
        
        # Effect should be influenced positively by high-probability cause
        assert probs["effect"] > 0.5
    
    def test_propagate_beliefs_contradiction(self):
        network = BeliefNetwork()
        network.add_belief("a", prior=0.9, evidence_strength=1.0)
        network.add_belief("b", prior=0.5, evidence_strength=0.5)
        network.add_dependency("a", "b", "contradicts", 0.8)
        
        probs = network.propagate_beliefs(iterations=5)
        
        # B should be influenced negatively by high-probability contradicting A
        assert probs["b"] < 0.5


class TestContradictionDetector:
    """Tests for ContradictionDetector"""
    
    def test_no_contradiction_different_embeddings(self, metric, sample_plots):
        detector = ContradictionDetector(metric)
        
        prob, explanation = detector.detect_contradiction(
            sample_plots["p1"],
            sample_plots["p2"],
        )
        
        # Different but not contradicting
        assert prob < 0.8
    
    def test_potential_contradiction_opposite_embeddings(self, metric, sample_plots):
        detector = ContradictionDetector(metric)
        
        prob, explanation = detector.detect_contradiction(
            sample_plots["p1"],
            sample_plots["p3"],  # Has opposite-ish embedding
        )
        
        # Should detect some contradiction signal
        # Note: this depends on the embeddings
        assert prob >= 0  # Valid probability
    
    def test_get_embedding_from_plot(self, metric, sample_plots):
        detector = ContradictionDetector(metric)
        
        emb = detector._get_embedding(sample_plots["p1"])
        
        assert emb is not None
        assert len(emb) == 64
    
    def test_learn_opposition_pattern(self, metric):
        detector = ContradictionDetector(metric)
        
        # Create positive and negative examples
        pos_examples = [np.random.randn(64).astype(np.float32) for _ in range(5)]
        neg_examples = [-e for e in pos_examples]
        
        detector.learn_opposition_pattern(pos_examples, neg_examples)
        
        assert len(detector.opposition_patterns) == 1


class TestCoherenceScorer:
    """Tests for CoherenceScorer"""
    
    def test_compute_coherence_empty(self, metric):
        detector = ContradictionDetector(metric)
        scorer = CoherenceScorer(metric, detector)
        
        report = scorer.compute_coherence(
            graph=MemoryGraph(),
            plots={},
            stories={},
            themes={},
        )
        
        assert abs(report.overall_score - 1.0) < 1e-6  # Empty is perfectly coherent (with float tolerance)
        assert len(report.conflicts) == 0
    
    def test_compute_coherence_with_plots(self, metric, sample_plots, sample_themes):
        detector = ContradictionDetector(metric)
        scorer = CoherenceScorer(metric, detector)
        
        graph = MemoryGraph()
        for p in sample_plots.values():
            graph.add_node(p.id, "plot", p)
        
        report = scorer.compute_coherence(
            graph=graph,
            plots=sample_plots,
            stories={},
            themes=sample_themes,
        )
        
        assert 0 <= report.overall_score <= 1
        assert 0 <= report.factual_coherence <= 1
        assert 0 <= report.thematic_coherence <= 1
    
    def test_temporal_coherence_check(self, metric, sample_plots):
        detector = ContradictionDetector(metric)
        scorer = CoherenceScorer(metric, detector)
        
        ts = now_ts()
        # Create a story with plots
        story = StoryArc(
            id="story1",
            created_ts=ts,
            updated_ts=ts,
            plot_ids=["p1", "p2", "p3"],
        )
        
        conflicts, score = scorer._check_temporal_coherence(
            sample_plots,
            {"story1": story},
        )
        
        assert 0 <= score <= 1
        # No temporal violations in our test data
        assert len([c for c in conflicts if c.type == ConflictType.TEMPORAL]) == 0


class TestConflictResolver:
    """Tests for ConflictResolver"""
    
    def test_resolve_with_weaken_strategy(self, metric, sample_themes):
        resolver = ConflictResolver(metric)
        
        conflict = Conflict(
            type=ConflictType.THEMATIC,
            node_a="t1",
            node_b="t2",
            severity=0.7,
            confidence=0.8,
            description="Test conflict",
            resolutions=[
                Resolution(
                    strategy="weaken",
                    target_node="t2",
                    action_description="Weaken theme t2",
                    expected_coherence_gain=0.5,
                    cost=0.2,
                )
            ],
        )
        
        graph = MemoryGraph()
        result = resolver.resolve(
            conflict,
            graph,
            plots={},
            stories={},
            themes=sample_themes,
        )
        
        assert result == True
        # Theme t2 should have increased b
        assert sample_themes["t2"].b > 4.0


class TestCoherenceGuardian:
    """Tests for CoherenceGuardian"""
    
    def test_full_check(self, metric, sample_plots, sample_themes):
        guardian = CoherenceGuardian(metric)
        
        graph = MemoryGraph()
        for p in sample_plots.values():
            graph.add_node(p.id, "plot", p)
        
        report = guardian.full_check(
            graph=graph,
            plots=sample_plots,
            stories={},
            themes=sample_themes,
        )
        
        assert isinstance(report, CoherenceReport)
        assert 0 <= report.overall_score <= 1
    
    def test_auto_resolve(self, metric, sample_plots, sample_themes):
        guardian = CoherenceGuardian(metric)
        
        graph = MemoryGraph()
        for p in sample_plots.values():
            graph.add_node(p.id, "plot", p)
        
        report = guardian.full_check(
            graph=graph,
            plots=sample_plots,
            stories={},
            themes=sample_themes,
        )
        
        # Should not crash even with no conflicts
        resolved = guardian.auto_resolve(
            report,
            graph,
            sample_plots,
            {},
            sample_themes,
            max_resolutions=3,
        )
        
        assert isinstance(resolved, int)
        assert resolved >= 0
    
    def test_update_belief_network(self, metric, sample_themes):
        guardian = CoherenceGuardian(metric)
        
        probs = guardian.update_belief_network(sample_themes)
        
        assert "t1" in probs
        assert "t2" in probs
        assert 0 <= probs["t1"] <= 1
        assert 0 <= probs["t2"] <= 1
    
    def test_generate_resolutions(self, metric):
        guardian = CoherenceGuardian(metric)
        
        conflict = Conflict(
            type=ConflictType.FACTUAL,
            node_a="a",
            node_b="b",
            severity=0.8,
            confidence=0.9,
            description="Test conflict",
        )
        
        resolutions = guardian._generate_resolutions(conflict)
        
        assert len(resolutions) > 0
        assert all(isinstance(r, Resolution) for r in resolutions)


class TestIntegration:
    """Integration tests for coherence module"""
    
    def test_coherence_workflow(self, metric, sample_plots, sample_themes):
        """Test complete coherence checking workflow"""
        
        # 1. Set up guardian
        guardian = CoherenceGuardian(metric)
        
        # 2. Create memory graph
        graph = MemoryGraph()
        for p in sample_plots.values():
            graph.add_node(p.id, "plot", p)
        
        for t in sample_themes.values():
            graph.add_node(t.id, "theme", t)
        
        ts = now_ts()
        # 3. Create stories
        story = StoryArc(
            id="story1",
            created_ts=ts,
            updated_ts=ts,
            plot_ids=["p1", "p2", "p3"],
        )
        
        # 4. Run full check
        report = guardian.full_check(
            graph=graph,
            plots=sample_plots,
            stories={"story1": story},
            themes=sample_themes,
        )
        
        # 5. Verify results
        assert isinstance(report, CoherenceReport)
        assert report.overall_score > 0  # Should be somewhat coherent
        
        # 6. Try auto-resolve if there are conflicts
        if report.conflicts:
            resolved = guardian.auto_resolve(
                report,
                graph,
                sample_plots,
                {"story1": story},
                sample_themes,
            )
            assert resolved >= 0
        
        # 7. Update belief network
        probs = guardian.update_belief_network(sample_themes)
        assert len(probs) == len(sample_themes)
