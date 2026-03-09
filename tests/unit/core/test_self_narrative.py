"""
AURORA 自我叙事模块的测试
"""

from __future__ import annotations

import numpy as np
import pytest

from aurora.core.models.plot import Plot
from aurora.core.models.theme import Theme
from aurora.core.components.metric import LowRankMetric
from aurora.core.personality import load_personality_profile
from aurora.core.self_narrative import (
    CapabilityBelief,
    RelationshipBelief,
    SelfNarrative,
    SelfNarrativeEngine,
    IdentityTracker,
)
from aurora.utils.time_utils import now_ts


@pytest.fixture
def metric():
    return LowRankMetric(dim=64, rank=16, seed=42)


@pytest.fixture
def sample_themes():
    """Create sample themes for testing"""
    ts = now_ts()
    return [
        Theme(
            id="cap_coding",
            created_ts=ts,
            updated_ts=ts,
            name="编程能力",
            description="能够编写和解释代码",
            theme_type="capability",
            a=8.0,
            b=2.0,
        ),
        Theme(
            id="cap_analysis",
            created_ts=ts,
            updated_ts=ts,
            name="分析能力",
            description="能够分析问题和数据",
            theme_type="capability",
            a=7.0,
            b=3.0,
        ),
        Theme(
            id="lim_realtime",
            created_ts=ts,
            updated_ts=ts,
            name="实时信息限制",
            description="无法获取实时信息",
            theme_type="limitation",
            a=9.0,
            b=1.0,
        ),
    ]


@pytest.fixture
def sample_plot():
    emb = np.random.randn(64).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    
    return Plot(
        id="plot1",
        text="User asked about code optimization",
        embedding=emb,
        ts=now_ts(),
        actors=["user"],
    )


class TestCapabilityBelief:
    """Tests for CapabilityBelief"""
    
    def test_initial_probability(self):
        cap = CapabilityBelief(name="test", description="Test capability")
        assert abs(cap.capability_probability() - 0.5) < 0.01
    
    def test_update_success(self):
        cap = CapabilityBelief(name="test", description="Test capability")
        
        for _ in range(10):
            cap.update(success=True)
        
        assert cap.capability_probability() > 0.8
        assert cap.success_count == 10
    
    def test_update_failure(self):
        cap = CapabilityBelief(name="test", description="Test capability")
        
        for _ in range(10):
            cap.update(success=False)
        
        assert cap.capability_probability() < 0.2
        assert cap.failure_count == 10
    
    def test_update_with_context(self):
        cap = CapabilityBelief(name="test", description="Test capability")
        
        cap.update(success=True, context="编程")
        cap.update(success=False, context="实时数据")
        
        assert "编程" in cap.positive_contexts
        assert "实时数据" in cap.negative_contexts
    
    def test_to_narrative_high_capability(self):
        cap = CapabilityBelief(
            name="编程",
            description="编写代码",
            a=9.0,
            b=1.0,
        )
        
        narrative = cap.to_narrative()
        
        assert "擅长" in narrative
    
    def test_to_narrative_low_capability(self):
        cap = CapabilityBelief(
            name="预测",
            description="预测未来",
            a=1.0,
            b=9.0,
        )
        
        narrative = cap.to_narrative()
        
        assert "困难" in narrative or "学习" in narrative
    
    def test_sample(self):
        cap = CapabilityBelief(name="test", description="Test", a=9.0, b=1.0)
        rng = np.random.default_rng(42)
        
        # Should mostly return True with high a
        samples = [cap.sample(rng) for _ in range(100)]
        assert sum(samples) > 80


class TestRelationshipBelief:
    """Tests for RelationshipBelief"""
    
    def test_initial_values(self):
        rel = RelationshipBelief(entity_id="user1", entity_type="user")
        
        assert abs(rel.trust() - 0.5) < 0.01
        assert abs(rel.familiarity() - 0.5) < 0.01
    
    def test_update_positive_interaction(self):
        rel = RelationshipBelief(entity_id="user1", entity_type="user")
        
        for _ in range(10):
            rel.update_interaction(positive=True)
        
        assert rel.trust() > 0.8
        assert rel.familiarity() > 0.5
        assert rel.positive_interactions == 10
    
    def test_update_negative_interaction(self):
        rel = RelationshipBelief(entity_id="user1", entity_type="user")
        
        for _ in range(10):
            rel.update_interaction(positive=False)
        
        assert rel.trust() < 0.5
        assert rel.negative_interactions == 10
    
    def test_update_preference(self):
        rel = RelationshipBelief(entity_id="user1", entity_type="user")
        
        rel.update_preference("简洁", 0.8)
        rel.update_preference("详细", 0.3)
        
        assert "简洁" in rel.preferences
        assert rel.preferences["简洁"] == 0.8
    
    def test_update_preference_average(self):
        rel = RelationshipBelief(entity_id="user1", entity_type="user")
        
        rel.update_preference("简洁", 1.0)
        rel.update_preference("简洁", 0.0)
        
        # Exponential moving average
        assert 0 < rel.preferences["简洁"] < 1.0
    
    def test_to_narrative(self):
        rel = RelationshipBelief(entity_id="user1", entity_type="user")
        
        for _ in range(10):
            rel.update_interaction(positive=True)
        
        rel.update_preference("代码", 0.9)
        
        narrative = rel.to_narrative()
        
        assert len(narrative) > 0
        assert "信任" in narrative or "关系" in narrative


class TestSelfNarrative:
    """Tests for SelfNarrative"""
    
    def test_default_values(self):
        narrative = SelfNarrative()
        
        assert "AI助手" in narrative.identity_statement
        assert len(narrative.core_values) > 0
    
    def test_get_capability(self):
        narrative = SelfNarrative()
        
        cap = narrative.get_capability("编程")
        
        assert cap.name == "编程"
        assert "编程" in narrative.capabilities
    
    def test_get_relationship(self):
        narrative = SelfNarrative()
        
        rel = narrative.get_relationship("user1")
        
        assert rel.entity_id == "user1"
        assert "user1" in narrative.relationships
    
    def test_to_full_narrative(self):
        narrative = SelfNarrative()
        
        # Add some data
        narrative.get_capability("编程").update(success=True)
        narrative.get_relationship("user1").update_interaction(positive=True)
        
        full = narrative.to_full_narrative()
        
        assert "## 我是谁" in full
        assert "## 核心价值" in full
    
    def test_log_evolution(self):
        narrative = SelfNarrative()
        
        narrative.log_evolution("test", "Test change")
        
        assert len(narrative.evolution_log) == 1
        assert narrative.evolution_log[0][1] == "test"
    
    def test_log_evolution_bounded(self):
        narrative = SelfNarrative()
        
        for i in range(150):
            narrative.log_evolution("test", f"Change {i}")
        
        assert len(narrative.evolution_log) == 100


class TestSelfNarrativeEngine:
    """Tests for SelfNarrativeEngine"""
    
    def test_update_from_themes(self, metric, sample_themes):
        engine = SelfNarrativeEngine(metric, profile=load_personality_profile("aurora-v2-native"))
        
        changed = engine.update_from_themes(sample_themes)
        
        # Should have updated capabilities
        assert len(engine.narrative.capabilities) > 0
    
    def test_update_from_interaction(self, metric, sample_plot):
        engine = SelfNarrativeEngine(metric, profile=load_personality_profile("aurora-v2-native"))
        
        engine.update_from_interaction(
            plot=sample_plot,
            success=True,
            entity_id="user1",
        )
        
        # Should have relationship with user1
        assert "user1" in engine.narrative.relationships
    
    def test_add_and_resolve_tension(self, metric):
        engine = SelfNarrativeEngine(metric, profile=load_personality_profile("aurora-v2-native"))
        
        engine.add_tension("需要实时数据但无法获取")
        
        assert "需要实时数据但无法获取" in engine.narrative.unresolved_tensions
        
        engine.resolve_tension("需要实时数据但无法获取")
        
        assert "需要实时数据但无法获取" not in engine.narrative.unresolved_tensions
    
    def test_check_coherence(self, metric):
        engine = SelfNarrativeEngine(metric, profile=load_personality_profile("aurora-v2-native"))
        
        # Add some capabilities
        engine.narrative.get_capability("test").update(success=True)
        
        score = engine.check_coherence()
        
        assert 0 <= score <= 1
    
    def test_momentum_slows_changes(self, metric):
        engine = SelfNarrativeEngine(metric, profile=load_personality_profile("aurora-v2-native"))
        engine.momentum = 0.99  # Very high momentum
        
        ts = now_ts()
        # Create a high-confidence theme
        theme = Theme(
            id="t1",
            created_ts=ts,
            updated_ts=ts,
            name="test_cap",
            description="Test capability",
            theme_type="capability",
            a=100.0,
            b=1.0,
        )
        
        # Get initial capability probability
        cap = engine.narrative.get_capability("test_cap")
        initial_prob = cap.capability_probability()
        
        engine._update_capability_from_theme(theme)
        
        # With high momentum, change should be small
        final_prob = cap.capability_probability()
        assert abs(final_prob - initial_prob) < 0.5


class TestIdentityTracker:
    """Tests for IdentityTracker"""
    
    def test_snapshot(self):
        tracker = IdentityTracker()
        narrative = SelfNarrative()
        
        tracker.snapshot(narrative)
        
        assert len(tracker.snapshots) == 1
    
    def test_snapshot_bounded(self):
        tracker = IdentityTracker()
        narrative = SelfNarrative()
        
        for _ in range(150):
            tracker.snapshot(narrative)
        
        assert len(tracker.snapshots) == 100
    
    def test_detect_drift_no_change(self):
        tracker = IdentityTracker()
        narrative = SelfNarrative()
        
        tracker.snapshot(narrative)
        tracker.snapshot(narrative)
        
        drift = tracker.detect_drift(narrative)
        
        # No change = no drift
        assert drift < 0.1
    
    def test_log_change(self):
        tracker = IdentityTracker()
        
        tracker.log_change("capability_update", 0.3)
        
        assert len(tracker.change_events) == 1
    
    def test_stability_score_no_changes(self):
        tracker = IdentityTracker()
        
        score = tracker.get_stability_score()
        
        # No changes = perfect stability
        assert score == 1.0
    
    def test_stability_score_with_changes(self):
        tracker = IdentityTracker()
        
        for _ in range(10):
            tracker.log_change("update", 0.5)
        
        score = tracker.get_stability_score()
        
        # Many changes = lower stability
        assert score < 1.0


class TestIntegration:
    """Integration tests for self-narrative module"""
    
    def test_complete_workflow(self, metric, sample_themes, sample_plot):
        """Test complete self-narrative workflow"""
        
        # 1. Create engine
        engine = SelfNarrativeEngine(metric, profile=load_personality_profile("aurora-v2-native"))
        tracker = IdentityTracker()
        
        # 2. Initial snapshot
        tracker.snapshot(engine.narrative)
        
        # 3. Update from themes
        engine.update_from_themes(sample_themes)
        
        # 4. Record interactions
        for i in range(5):
            engine.update_from_interaction(
                plot=sample_plot,
                success=True,
                entity_id=f"user_{i % 2}",
            )
        
        # 5. Add tensions
        engine.add_tension("有时无法获取实时信息")
        
        # 6. Check coherence
        coherence = engine.check_coherence()
        assert 0 <= coherence <= 1
        
        # 7. Take another snapshot
        tracker.snapshot(engine.narrative)
        
        # 8. Check for drift
        drift = tracker.detect_drift(engine.narrative)
        assert 0 <= drift <= 1
        
        # 9. Generate full narrative
        full = engine.narrative.to_full_narrative()
        assert len(full) > 0
        assert "我是谁" in full
        
        # 10. Check stability
        stability = tracker.get_stability_score()
        assert 0 <= stability <= 1
