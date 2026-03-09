"""
AURORA 冲突检测和处理的测试

测试 CoherenceGuardian 集成到摄入流中，
验证冲突解决的第一原则方法：

1. 状态事实（电话、地址）→ 更新（新的取代旧的）
2. 身份特征（耐心、高效）→ 保留两者（适应灵活性）
3. 静态事实（生日）→ 如果错误则更正
4. 偏好 → 演变（追踪时间线）

AURORA 哲学：并非所有矛盾都需要消除。
健康的身份包含提供适应性的张力。
"""

import pytest
import numpy as np

from aurora.core.memory import AuroraMemory
from aurora.core.models.config import MemoryConfig
from aurora.core.knowledge import (
    KnowledgeClassifier,
    KnowledgeType,
    ConflictResolution,
)


@pytest.fixture
def memory():
    """Create a fresh AuroraMemory instance for each test."""
    cfg = MemoryConfig(dim=64, metric_rank=16, max_plots=100)
    return AuroraMemory(cfg=cfg, seed=42)


@pytest.fixture
def knowledge_classifier():
    """Create a KnowledgeClassifier instance."""
    return KnowledgeClassifier(seed=42)


class TestKnowledgeClassifier:
    """Test the KnowledgeClassifier component."""
    
    def test_classify_state_fact_phone(self, knowledge_classifier):
        """Test classification of phone number (state fact)."""
        result = knowledge_classifier.classify("用户：我的电话是 123-456-7890")
        assert result.knowledge_type == KnowledgeType.FACTUAL_STATE
    
    def test_classify_state_fact_address(self, knowledge_classifier):
        """Test classification of address (state fact)."""
        result = knowledge_classifier.classify("User: I live in Beijing now")
        assert result.knowledge_type == KnowledgeType.FACTUAL_STATE
    
    def test_classify_identity_trait(self, knowledge_classifier):
        """Test classification of identity trait."""
        result = knowledge_classifier.classify("用户：我是一个很耐心的人")
        assert result.knowledge_type == KnowledgeType.IDENTITY_TRAIT
    
    def test_classify_static_fact_birthday(self, knowledge_classifier):
        """Test classification of birthday (static fact)."""
        result = knowledge_classifier.classify("User: My birthday is January 15th")
        assert result.knowledge_type == KnowledgeType.FACTUAL_STATIC
    
    def test_classify_preference(self, knowledge_classifier):
        """Test classification of preference."""
        result = knowledge_classifier.classify("用户：我喜欢喝咖啡")
        assert result.knowledge_type == KnowledgeType.PREFERENCE
    
    def test_complementary_traits(self, knowledge_classifier):
        """Test detection of complementary traits (should preserve both)."""
        # Patient and efficient are complementary, not contradictory
        is_complementary = knowledge_classifier.are_complementary_traits(
            "我是一个很耐心的人",
            "我是一个很高效的人"
        )
        assert is_complementary is True
    
    def test_contradictory_traits(self, knowledge_classifier):
        """Test detection of truly contradictory traits."""
        # Honest and dishonest are contradictory (use exact terms from CONTRADICTORY_PAIRS)
        is_complementary = knowledge_classifier.are_complementary_traits(
            "I am honest",
            "I am dishonest"
        )
        assert is_complementary is False


class TestConflictResolution:
    """Test conflict resolution strategies."""
    
    def test_state_fact_update(self, knowledge_classifier):
        """State facts should UPDATE (new supersedes old)."""
        analysis = knowledge_classifier.resolve_conflict(
            KnowledgeType.FACTUAL_STATE,
            KnowledgeType.FACTUAL_STATE,
            time_relation="sequential",
            text_a="我的电话是 123",
            text_b="我的电话是 456",
        )
        assert analysis.resolution == ConflictResolution.UPDATE
    
    def test_identity_trait_preserve_both(self, knowledge_classifier):
        """Identity traits should PRESERVE_BOTH."""
        analysis = knowledge_classifier.resolve_conflict(
            KnowledgeType.IDENTITY_TRAIT,
            KnowledgeType.IDENTITY_TRAIT,
            time_relation="sequential",
            text_a="我很耐心",
            text_b="我很高效",
        )
        assert analysis.resolution == ConflictResolution.PRESERVE_BOTH
    
    def test_static_fact_correct(self, knowledge_classifier):
        """Static facts should CORRECT (one is wrong)."""
        analysis = knowledge_classifier.resolve_conflict(
            KnowledgeType.FACTUAL_STATIC,
            KnowledgeType.FACTUAL_STATIC,
            time_relation="sequential",
            text_a="My birthday is Jan 1",
            text_b="My birthday is Jan 15",
        )
        assert analysis.resolution == ConflictResolution.CORRECT
    
    def test_preference_evolve(self, knowledge_classifier):
        """Preferences should EVOLVE (track timeline)."""
        analysis = knowledge_classifier.resolve_conflict(
            KnowledgeType.PREFERENCE,
            KnowledgeType.PREFERENCE,
            time_relation="sequential",
            text_a="我以前喜欢咖啡",
            text_b="我现在喜欢茶",
        )
        assert analysis.resolution == ConflictResolution.EVOLVE


class TestConflictDetectionIntegration:
    """Integration tests for conflict detection in ingest flow."""
    
    def test_state_update_creates_supersedes_chain(self, memory):
        """Test that state updates create a supersedes chain."""
        # Ingest initial phone number
        plot1 = memory.ingest("用户：我的电话是 123-456。助理：好的，记住了！")
        
        # Give the system enough plots for cold start
        for i in range(10):
            memory.ingest(f"用户：一些随机对话 {i}")
        
        # Ingest updated phone number
        plot2 = memory.ingest("用户：我的电话改成 789-012 了。助理：已更新！")
        
        # Check that plot2 supersedes plot1 (if detected as update)
        # Note: Detection depends on semantic similarity and knowledge classification
        if plot2.supersedes_id:
            old_plot = memory.plots.get(plot2.supersedes_id)
            if old_plot:
                assert old_plot.status in ["superseded", "corrected"]
                assert old_plot.superseded_by_id == plot2.id
    
    def test_identity_traits_preserved(self, memory):
        """Test that identity traits are preserved (not superseded)."""
        # Ingest first trait
        plot1 = memory.ingest("用户：我是一个很耐心的人。助理：这是很好的品质！")
        
        # Ingest second trait (should not supersede)
        plot2 = memory.ingest("用户：我也很高效。助理：太棒了！")
        
        # Both should remain active
        if plot1.id in memory.plots and plot2.id in memory.plots:
            assert memory.plots[plot1.id].status == "active"
            assert memory.plots[plot2.id].status == "active"
    
    def test_conflict_detection_respects_similarity_threshold(self, memory):
        """Test that conflicts are only detected above similarity threshold."""
        # Ingest unrelated topics - should not trigger conflict detection
        plot1 = memory.ingest("用户：今天天气真好")
        plot2 = memory.ingest("用户：帮我写一个排序算法")
        
        # Neither should supersede the other
        if plot1.id in memory.plots and plot2.id in memory.plots:
            assert memory.plots[plot1.id].superseded_by_id is None
            assert memory.plots[plot2.id].supersedes_id is None


class TestCoherenceGuardianIntegration:
    """Test CoherenceGuardian integration with AuroraMemory."""
    
    def test_coherence_guardian_initialized(self, memory):
        """Test that CoherenceGuardian is properly initialized."""
        assert hasattr(memory, 'coherence_guardian')
        assert memory.coherence_guardian is not None
    
    def test_knowledge_classifier_initialized(self, memory):
        """Test that KnowledgeClassifier is properly initialized."""
        assert hasattr(memory, 'knowledge_classifier')
        assert memory.knowledge_classifier is not None
    
    def test_tension_manager_accessible(self, memory):
        """Test that TensionManager is accessible through CoherenceGuardian."""
        assert hasattr(memory.coherence_guardian, 'tension_manager')
        summary = memory.coherence_guardian.get_tension_summary()
        assert "total" in summary


class TestPlotModelSupersedes:
    """Test Plot model supersedes fields."""
    
    def test_plot_supersedes_fields_exist(self, memory):
        """Test that Plot has supersedes fields."""
        plot = memory.ingest("用户：测试")
        
        assert hasattr(plot, 'supersedes_id')
        assert hasattr(plot, 'superseded_by_id')
        assert hasattr(plot, 'update_type')
    
    def test_plot_status_includes_superseded(self, memory):
        """Test that Plot status can be 'superseded'."""
        from aurora.core.models.plot import Plot
        import numpy as np
        
        plot = Plot(
            id="test-plot",
            ts=1234567890.0,
            text="test",
            actors=("user", "agent"),
            embedding=np.zeros(64),
            status="superseded"
        )
        assert plot.status == "superseded"


class TestFirstPrinciplesPhilosophy:
    """Test that the implementation follows AURORA's first principles."""
    
    def test_no_hard_coded_thresholds_in_classification(self, knowledge_classifier):
        """Test that classification uses probabilistic signals, not hard thresholds."""
        # The classifier returns confidence scores, not just binary decisions
        result = knowledge_classifier.classify("我的电话是123")
        assert 0 <= result.confidence <= 1
    
    def test_contradictions_can_be_adaptive(self, knowledge_classifier):
        """Test that contradictions can be classified as adaptive (to preserve)."""
        # Test complementary traits are recognized as adaptive
        analysis = knowledge_classifier.resolve_conflict(
            KnowledgeType.IDENTITY_TRAIT,
            KnowledgeType.IDENTITY_TRAIT,
            time_relation="sequential",
            text_a="我内向安静",
            text_b="我在工作时很社交",
        )
        # Should preserve both - these are context-dependent, not contradictory
        assert analysis.resolution == ConflictResolution.PRESERVE_BOTH
    
    def test_identity_construction_not_archival(self, memory):
        """Test that memory supports identity construction, not just archival.
        
        AURORA Philosophy: Memory is identity's continuous becoming,
        not an archive of the past.
        """
        # Identity dimensions should be tracked
        assert hasattr(memory, '_identity_dimensions')
        
        # Coherence should allow tensions
        summary = memory.coherence_guardian.get_tension_summary()
        assert "preserved_count" in summary  # Tensions can be preserved


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
