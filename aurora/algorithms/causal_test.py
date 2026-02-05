"""
Tests for AURORA Causal Inference Module
"""

from __future__ import annotations

import numpy as np
import pytest

from aurora.algorithms.models.plot import Plot
from aurora.algorithms.models.story import StoryArc
from aurora.algorithms.models.theme import Theme
from aurora.algorithms.graph.memory_graph import MemoryGraph
from aurora.algorithms.components.metric import LowRankMetric
from aurora.algorithms.causal import (
    CausalEdgeBelief,
    CausalDiscovery,
    InterventionEngine,
    CounterfactualReasoner,
    CausalMemoryGraph,
)
from aurora.utils.time_utils import now_ts


@pytest.fixture
def metric():
    return LowRankMetric(dim=64, rank=16, seed=42)


@pytest.fixture
def sample_plots(metric):
    """Create sample plots for testing"""
    plots = []
    base_time = now_ts()
    
    # Create a causal chain: A -> B -> C
    emb_a = np.random.randn(64).astype(np.float32)
    emb_a = emb_a / np.linalg.norm(emb_a)
    
    emb_b = 0.7 * emb_a + 0.3 * np.random.randn(64).astype(np.float32)
    emb_b = emb_b / np.linalg.norm(emb_b)
    
    emb_c = 0.6 * emb_b + 0.4 * np.random.randn(64).astype(np.float32)
    emb_c = emb_c / np.linalg.norm(emb_c)
    
    plots.append(Plot(
        id="plot_a",
        text="User asked about programming",
        embedding=emb_a,
        ts=base_time,
        actors=["user"],
    ))
    
    plots.append(Plot(
        id="plot_b", 
        text="Agent explained coding concepts",
        embedding=emb_b,
        ts=base_time + 60,
        actors=["agent"],
    ))
    
    plots.append(Plot(
        id="plot_c",
        text="User successfully implemented the code",
        embedding=emb_c,
        ts=base_time + 120,
        actors=["user"],
    ))
    
    return plots


class TestCausalEdgeBelief:
    """Tests for CausalEdgeBelief"""
    
    def test_initial_direction_belief_is_uniform(self):
        belief = CausalEdgeBelief(edge_type="causal")
        assert abs(belief.direction_belief() - 0.5) < 0.01
    
    def test_update_direction_evidence(self):
        belief = CausalEdgeBelief(edge_type="causal")
        
        # Add forward evidence
        for _ in range(10):
            belief.update_direction_evidence(forward=True)
        
        # Direction should now favor forward
        assert belief.direction_belief() > 0.8
    
    def test_causal_strength_default(self):
        belief = CausalEdgeBelief(edge_type="causal")
        assert abs(belief.causal_strength() - 0.5) < 0.01
    
    def test_effective_causal_weight(self):
        belief = CausalEdgeBelief(
            edge_type="causal",
            dir_a=9.0, dir_b=1.0,  # Strong forward direction
            str_a=8.0, str_b=2.0,  # Strong causal strength
            conf_a=1.0, conf_b=9.0,  # Low confounding
        )
        
        weight = belief.effective_causal_weight()
        # Should be high: direction * strength * (1 - confound)
        # ≈ 0.9 * 0.8 * 0.9 = 0.648
        assert weight > 0.5
    
    def test_intervention_evidence_reduces_confound_uncertainty(self):
        belief = CausalEdgeBelief(edge_type="causal")
        initial_conf_b = belief.conf_b
        
        belief.update_intervention_evidence(effect_observed=True)
        
        # Intervention should increase conf_b (reduce confound probability)
        assert belief.conf_b > initial_conf_b
        assert belief.intervention_count == 1


class TestCausalDiscovery:
    """Tests for CausalDiscovery"""
    
    def test_temporal_precedence(self, metric, sample_plots):
        discovery = CausalDiscovery(metric)
        
        p_forward, p_backward, p_confound = discovery.infer_direction(
            sample_plots[0],  # Earlier
            sample_plots[1],  # Later
            sample_plots,
        )
        
        # Should favor forward direction (earlier -> later)
        assert p_forward > p_backward
    
    def test_transfer_entropy_signal(self, metric, sample_plots):
        discovery = CausalDiscovery(metric)
        
        # B depends on A semantically
        te = discovery._estimate_transfer_entropy(
            sample_plots[0],  # A
            sample_plots[1],  # B
            sample_plots,
        )
        
        # Transfer entropy should be positive
        assert te >= 0
    
    def test_confound_detection(self, metric, sample_plots):
        discovery = CausalDiscovery(metric)
        
        p_forward, p_backward, p_confound = discovery.infer_direction(
            sample_plots[0],
            sample_plots[2],
            sample_plots,
        )
        
        # With an intermediate node (plot_b), there might be confound signals
        # This is a weak test - just checking it returns valid probability
        assert 0 <= p_confound <= 1


class TestInterventionEngine:
    """Tests for InterventionEngine"""
    
    def test_do_intervention_basic(self, metric, sample_plots):
        engine = InterventionEngine(metric)
        
        # Build a simple graph
        graph = MemoryGraph()
        for p in sample_plots:
            graph.add_node(p.id, "plot", p)
        
        # Add causal beliefs
        beliefs = {
            ("plot_a", "plot_b"): CausalEdgeBelief(edge_type="causal", dir_a=9.0, dir_b=1.0, str_a=8.0, str_b=2.0),
            ("plot_b", "plot_c"): CausalEdgeBelief(edge_type="causal", dir_a=9.0, dir_b=1.0, str_a=8.0, str_b=2.0),
        }
        
        # Intervene on plot_a
        new_embedding = np.random.randn(64).astype(np.float32)
        new_embedding = new_embedding / np.linalg.norm(new_embedding)
        
        result = engine.do_intervention(
            graph=graph,
            target_id="plot_a",
            intervention_value=new_embedding,
            causal_beliefs=beliefs,
        )
        
        assert result.target_id == "plot_a"
        assert "plot_a" in result.predicted_effects
        assert result.confidence > 0
    
    def test_causal_dag_construction(self, metric, sample_plots):
        engine = InterventionEngine(metric)
        
        graph = MemoryGraph()
        for p in sample_plots:
            graph.add_node(p.id, "plot", p)
        
        beliefs = {
            ("plot_a", "plot_b"): CausalEdgeBelief(edge_type="causal", dir_a=9.0, dir_b=1.0),
            ("plot_b", "plot_c"): CausalEdgeBelief(edge_type="causal", dir_a=9.0, dir_b=1.0),
        }
        
        dag = engine._build_causal_dag(graph, beliefs)
        
        # Should have edges A->B and B->C
        assert dag.has_edge("plot_a", "plot_b")
        assert dag.has_edge("plot_b", "plot_c")


class TestCounterfactualReasoner:
    """Tests for CounterfactualReasoner"""
    
    def test_counterfactual_query(self, metric, sample_plots):
        intervention = InterventionEngine(metric)
        reasoner = CounterfactualReasoner(metric, intervention)
        
        graph = MemoryGraph()
        for p in sample_plots:
            graph.add_node(p.id, "plot", p)
        
        beliefs = {
            ("plot_a", "plot_b"): CausalEdgeBelief(edge_type="causal", dir_a=9.0, dir_b=1.0, str_a=8.0, str_b=2.0),
            ("plot_b", "plot_c"): CausalEdgeBelief(edge_type="causal", dir_a=9.0, dir_b=1.0, str_a=8.0, str_b=2.0),
        }
        
        factual_values = {
            p.id: p.embedding for p in sample_plots
        }
        
        # Ask: what if plot_a had been different?
        cf_value = np.random.randn(64).astype(np.float32)
        cf_value = cf_value / np.linalg.norm(cf_value)
        
        result = reasoner.query(
            graph=graph,
            causal_beliefs=beliefs,
            factual_values=factual_values,
            antecedent_id="plot_a",
            antecedent_cf_value=cf_value,
            query_id="plot_c",
        )
        
        assert result.query_variable == "plot_c"
        assert result.difference_magnitude >= 0
        assert len(result.explanation) > 0


class TestCausalMemoryGraph:
    """Tests for CausalMemoryGraph"""
    
    def test_add_causal_edge(self, metric):
        graph = CausalMemoryGraph(metric)
        
        belief = graph.add_causal_edge("node_a", "node_b", initial_direction_belief=0.7)
        
        assert ("node_a", "node_b") in graph.causal_beliefs
        assert belief.direction_belief() > 0.5
    
    def test_get_causal_ancestors(self, metric, sample_plots):
        graph = CausalMemoryGraph(metric)
        
        # Register nodes
        for p in sample_plots:
            graph.add_node(p.id, "plot", p)
        
        # Add causal edges: A -> B -> C
        graph.add_causal_edge("plot_a", "plot_b", 0.9)
        graph.add_causal_edge("plot_b", "plot_c", 0.9)
        
        # Update beliefs to be strong
        graph.causal_beliefs[("plot_a", "plot_b")].str_a = 8.0
        graph.causal_beliefs[("plot_b", "plot_c")].str_a = 8.0
        
        # Get ancestors of C
        ancestors = graph.get_causal_ancestors("plot_c")
        
        # Should include B and A
        ancestor_ids = [a[0] for a in ancestors]
        assert "plot_b" in ancestor_ids
    
    def test_get_causal_descendants(self, metric, sample_plots):
        graph = CausalMemoryGraph(metric)
        
        for p in sample_plots:
            graph.add_node(p.id, "plot", p)
        
        graph.add_causal_edge("plot_a", "plot_b", 0.9)
        graph.add_causal_edge("plot_b", "plot_c", 0.9)
        
        graph.causal_beliefs[("plot_a", "plot_b")].str_a = 8.0
        graph.causal_beliefs[("plot_b", "plot_c")].str_a = 8.0
        
        # Get descendants of A
        descendants = graph.get_causal_descendants("plot_a")
        
        descendant_ids = [d[0] for d in descendants]
        assert "plot_b" in descendant_ids


class TestIntegration:
    """Integration tests"""
    
    def test_full_causal_workflow(self, metric, sample_plots):
        """Test complete causal inference workflow"""
        
        # 1. Create causal graph
        graph = CausalMemoryGraph(metric)
        
        for p in sample_plots:
            graph.add_node(p.id, "plot", p)
        
        # 2. Discover causal relations
        discovery = CausalDiscovery(metric)
        
        for i in range(len(sample_plots) - 1):
            p1 = sample_plots[i]
            p2 = sample_plots[i + 1]
            
            graph.infer_and_add_causal_edge(p1, p2, sample_plots)
        
        # 3. Verify structure
        assert len(graph.causal_beliefs) >= 2
        
        # 4. Do intervention
        new_emb = np.random.randn(64).astype(np.float32)
        new_emb = new_emb / np.linalg.norm(new_emb)
        
        result = graph.do("plot_a", new_emb)
        
        assert result.confidence > 0
        
        # 5. Get causal chain
        ancestors = graph.get_causal_ancestors("plot_c")
        descendants = graph.get_causal_descendants("plot_a")
        
        # Should have valid results
        assert isinstance(ancestors, list)
        assert isinstance(descendants, list)
