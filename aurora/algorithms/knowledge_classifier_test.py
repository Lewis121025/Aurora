"""
AURORA 知识类型分类器的测试
==========================================

测试基于第一原则的知识分类和冲突解决：
- 并非所有矛盾都需要消除
- 不同的知识类型需要不同的处理策略
- 互补的特征应该被保留，而不是解决
"""

import pytest
import numpy as np

from aurora.algorithms.knowledge_classifier import (
    KnowledgeClassifier,
    KnowledgeType,
    ConflictResolution,
    ClassificationResult,
    ConflictAnalysis,
    classify_knowledge,
    resolve_knowledge_conflict,
)


# =============================================================================
# 夹具
# =============================================================================

@pytest.fixture
def classifier():
    """创建具有固定种子的知识分类器以实现可重复性。"""
    return KnowledgeClassifier(seed=42)


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing."""
    rng = np.random.default_rng(42)
    return {
        "patient": rng.random(128).astype(np.float32),
        "efficient": rng.random(128).astype(np.float32),
        "honest": rng.random(128).astype(np.float32),
        "dishonest": -rng.random(128).astype(np.float32),  # Opposite direction
    }


# =============================================================================
# Test Knowledge Type Classification
# =============================================================================

class TestKnowledgeClassification:
    """Test suite for knowledge type classification."""
    
    def test_classify_factual_state_chinese(self, classifier):
        """Test classification of state facts in Chinese."""
        texts = [
            "用户住在北京",
            "他的电话是12345678",
            "我的邮箱是test@example.com",
            "他目前是产品经理",
            "她在谷歌工作",  # Using "在谷歌" pattern
        ]
        
        for text in texts:
            result = classifier.classify(text)
            assert result.knowledge_type == KnowledgeType.FACTUAL_STATE, \
                f"Expected FACTUAL_STATE for '{text}', got {result.knowledge_type}"
            assert result.confidence > 0.5
    
    def test_classify_factual_state_english(self, classifier):
        """Test classification of state facts in English."""
        texts = [
            "User lives in New York",
            "Her phone number is 555-1234",
            "He works at Microsoft",
            "Currently working as a developer",
        ]
        
        for text in texts:
            result = classifier.classify(text)
            assert result.knowledge_type == KnowledgeType.FACTUAL_STATE, \
                f"Expected FACTUAL_STATE for '{text}', got {result.knowledge_type}"
    
    def test_classify_factual_static_chinese(self, classifier):
        """Test classification of static facts in Chinese."""
        texts = [
            "他的生日是1月1日",
            "她出生于上海",
            "用户的出生日期是1990年5月",
        ]
        
        for text in texts:
            result = classifier.classify(text)
            assert result.knowledge_type == KnowledgeType.FACTUAL_STATIC, \
                f"Expected FACTUAL_STATIC for '{text}', got {result.knowledge_type}"
    
    def test_classify_factual_static_english(self, classifier):
        """Test classification of static facts in English."""
        texts = [
            "His birthday is January 1st",
            "She was born in London",
            "User's birthplace is Paris",
        ]
        
        for text in texts:
            result = classifier.classify(text)
            assert result.knowledge_type == KnowledgeType.FACTUAL_STATIC, \
                f"Expected FACTUAL_STATIC for '{text}', got {result.knowledge_type}"
    
    def test_classify_identity_trait_chinese(self, classifier):
        """Test classification of identity traits in Chinese."""
        texts = [
            "我是一个耐心的人",
            "他性格很外向",
            "她擅长分析问题",
            "我很认真负责",
        ]
        
        for text in texts:
            result = classifier.classify(text)
            assert result.knowledge_type == KnowledgeType.IDENTITY_TRAIT, \
                f"Expected IDENTITY_TRAIT for '{text}', got {result.knowledge_type}"
    
    def test_classify_identity_trait_english(self, classifier):
        """Test classification of identity traits in English."""
        texts = [
            "I am a patient person",
            "She is very efficient",
            "He tends to be analytical by nature",  # Added "by nature" for clearer trait
            "I am naturally curious",
        ]
        
        for text in texts:
            result = classifier.classify(text)
            assert result.knowledge_type == KnowledgeType.IDENTITY_TRAIT, \
                f"Expected IDENTITY_TRAIT for '{text}', got {result.knowledge_type}"
    
    def test_classify_preference_chinese(self, classifier):
        """Test classification of preferences in Chinese."""
        texts = [
            "我喜欢喝咖啡",
            "他偏好在早上工作",
            "她讨厌拥挤的地方",
            "我热爱阅读",
        ]
        
        for text in texts:
            result = classifier.classify(text)
            assert result.knowledge_type == KnowledgeType.PREFERENCE, \
                f"Expected PREFERENCE for '{text}', got {result.knowledge_type}"
    
    def test_classify_preference_english(self, classifier):
        """Test classification of preferences in English."""
        texts = [
            "I like coffee",
            "She prefers working remotely",
            "He loves hiking",
            "I enjoy reading books",
        ]
        
        for text in texts:
            result = classifier.classify(text)
            assert result.knowledge_type == KnowledgeType.PREFERENCE, \
                f"Expected PREFERENCE for '{text}', got {result.knowledge_type}"
    
    def test_classify_behavior_pattern_chinese(self, classifier):
        """Test classification of behavior patterns in Chinese."""
        texts = [
            "他通常早起",
            "她每天都锻炼",
            "我经常在晚上工作",
            "他总是准时",
        ]
        
        for text in texts:
            result = classifier.classify(text)
            assert result.knowledge_type == KnowledgeType.BEHAVIOR_PATTERN, \
                f"Expected BEHAVIOR_PATTERN for '{text}', got {result.knowledge_type}"
    
    def test_classify_behavior_pattern_english(self, classifier):
        """Test classification of behavior patterns in English."""
        texts = [
            "He usually wakes up early",
            "She exercises every day",
            "I often work late",
            "He always arrives on time",
        ]
        
        for text in texts:
            result = classifier.classify(text)
            assert result.knowledge_type == KnowledgeType.BEHAVIOR_PATTERN, \
                f"Expected BEHAVIOR_PATTERN for '{text}', got {result.knowledge_type}"
    
    def test_classify_unknown(self, classifier):
        """Test classification of ambiguous text."""
        texts = [
            "xyz123",
            "。。。",
            "",
        ]
        
        for text in texts:
            result = classifier.classify(text)
            assert result.knowledge_type == KnowledgeType.UNKNOWN, \
                f"Expected UNKNOWN for '{text}', got {result.knowledge_type}"
    
    def test_classification_result_structure(self, classifier):
        """Test that ClassificationResult has all expected fields."""
        result = classifier.classify("我住在北京")
        
        assert hasattr(result, 'knowledge_type')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'matched_patterns')
        assert isinstance(result.to_dict(), dict)


# =============================================================================
# Test Conflict Resolution
# =============================================================================

class TestConflictResolution:
    """Test suite for knowledge conflict resolution."""
    
    def test_resolve_state_update(self, classifier):
        """Test that state facts use UPDATE strategy."""
        analysis = classifier.resolve_conflict(
            type_a=KnowledgeType.FACTUAL_STATE,
            type_b=KnowledgeType.FACTUAL_STATE,
            time_relation="sequential",
            text_a="我住在北京",
            text_b="我住在上海",
        )
        
        assert analysis.resolution == ConflictResolution.UPDATE
        assert analysis.confidence > 0.7
    
    def test_resolve_static_fact_correction(self, classifier):
        """Test that static facts use CORRECT strategy."""
        analysis = classifier.resolve_conflict(
            type_a=KnowledgeType.FACTUAL_STATIC,
            type_b=KnowledgeType.FACTUAL_STATIC,
            time_relation="sequential",
            text_a="他的生日是1月1日",
            text_b="他的生日是2月2日",
        )
        
        assert analysis.resolution == ConflictResolution.CORRECT
        assert analysis.requires_human_review  # Static facts need verification
    
    def test_resolve_trait_preserve(self, classifier):
        """Test that traits use PRESERVE_BOTH by default."""
        analysis = classifier.resolve_conflict(
            type_a=KnowledgeType.IDENTITY_TRAIT,
            type_b=KnowledgeType.IDENTITY_TRAIT,
            time_relation="concurrent",
            text_a="我是一个耐心的人",
            text_b="我是一个高效的人",
        )
        
        assert analysis.resolution == ConflictResolution.PRESERVE_BOTH
        assert analysis.is_complementary  # Patient and efficient are complementary
    
    def test_resolve_preference_evolve(self, classifier):
        """Test that preferences use EVOLVE strategy."""
        analysis = classifier.resolve_conflict(
            type_a=KnowledgeType.PREFERENCE,
            type_b=KnowledgeType.PREFERENCE,
            time_relation="sequential",
            text_a="我喜欢喝咖啡",
            text_b="我喜欢喝茶",
        )
        
        assert analysis.resolution == ConflictResolution.EVOLVE
    
    def test_resolve_value_preserve(self, classifier):
        """Test that values use PRESERVE_BOTH strategy."""
        analysis = classifier.resolve_conflict(
            type_a=KnowledgeType.IDENTITY_VALUE,
            type_b=KnowledgeType.IDENTITY_VALUE,
            time_relation="concurrent",
        )
        
        assert analysis.resolution == ConflictResolution.PRESERVE_BOTH
    
    def test_resolve_cross_type_preserve(self, classifier):
        """Test that different types default to PRESERVE_BOTH."""
        analysis = classifier.resolve_conflict(
            type_a=KnowledgeType.FACTUAL_STATE,
            type_b=KnowledgeType.PREFERENCE,
            time_relation="concurrent",
        )
        
        # Different types can coexist
        assert analysis.resolution == ConflictResolution.PRESERVE_BOTH


# =============================================================================
# Test Complementary Trait Detection
# =============================================================================

class TestComplementaryTraits:
    """Test suite for complementary trait detection."""
    
    def test_patient_efficient_complementary(self, classifier):
        """Test that patient and efficient are recognized as complementary."""
        assert classifier.are_complementary_traits(
            "我是一个耐心的人",
            "我是一个高效的人",
        )
    
    def test_introverted_social_complementary(self, classifier):
        """Test that introverted and social are recognized as complementary."""
        assert classifier.are_complementary_traits(
            "我比较内向",
            "我善于社交",
        )
    
    def test_rational_emotional_complementary(self, classifier):
        """Test that rational and emotional are recognized as complementary."""
        assert classifier.are_complementary_traits(
            "我很理性",
            "我很感性",
        )
    
    def test_rigorous_flexible_complementary(self, classifier):
        """Test that rigorous and flexible are recognized as complementary."""
        assert classifier.are_complementary_traits(
            "我做事严谨",
            "我比较灵活",
        )
    
    def test_honest_dishonest_contradictory(self, classifier):
        """Test that honest and dishonest are recognized as contradictory."""
        assert not classifier.are_complementary_traits(
            "我很诚实",
            "我喜欢说谎",
        )
    
    def test_helpful_harmful_contradictory(self, classifier):
        """Test that helpful and harmful are recognized as contradictory."""
        assert not classifier.are_complementary_traits(
            "我乐于帮助别人",
            "我喜欢伤害别人",
        )
    
    def test_english_traits_complementary(self, classifier):
        """Test complementary trait detection in English."""
        assert classifier.are_complementary_traits(
            "I am patient",
            "I am efficient",
        )
        assert classifier.are_complementary_traits(
            "I am introverted",
            "I am social",
        )
    
    def test_english_traits_contradictory(self, classifier):
        """Test contradictory trait detection in English."""
        assert not classifier.are_complementary_traits(
            "I am honest",
            "I am a liar",
        )


# =============================================================================
# Test Convenience Functions
# =============================================================================

class TestConvenienceFunctions:
    """Test the convenience functions for one-off use."""
    
    def test_classify_knowledge_function(self):
        """Test the classify_knowledge convenience function."""
        result = classify_knowledge("我住在北京")
        assert result.knowledge_type == KnowledgeType.FACTUAL_STATE
    
    def test_resolve_knowledge_conflict_function(self):
        """Test the resolve_knowledge_conflict convenience function."""
        analysis = resolve_knowledge_conflict(
            text_a="我住在北京",
            text_b="我住在上海",
            time_relation="sequential",
        )
        assert analysis.resolution == ConflictResolution.UPDATE


# =============================================================================
# Test Serialization
# =============================================================================

class TestSerialization:
    """Test classifier state serialization."""
    
    def test_to_state_dict(self, classifier):
        """Test serialization to state dict."""
        # Classify some texts to build stats
        classifier.classify("我住在北京")
        classifier.classify("我喜欢咖啡")
        
        state = classifier.to_state_dict()
        
        assert "seed" in state
        assert "classification_stats" in state
    
    def test_from_state_dict(self, classifier):
        """Test restoration from state dict."""
        # Classify some texts
        classifier.classify("我住在北京")
        classifier.classify("我住在上海")
        
        # Save and restore
        state = classifier.to_state_dict()
        restored = KnowledgeClassifier.from_state_dict(state)
        
        assert restored._seed == classifier._seed
        assert restored._classification_stats == classifier._classification_stats
    
    def test_get_statistics(self, classifier):
        """Test getting classification statistics."""
        classifier.classify("我住在北京")  # FACTUAL_STATE
        classifier.classify("我喜欢咖啡")  # PREFERENCE
        
        stats = classifier.get_statistics()
        
        assert "total_classifications" in stats
        assert stats["total_classifications"] == 2
        assert "by_type" in stats
        assert "distribution" in stats


# =============================================================================
# Test Batch Classification
# =============================================================================

class TestBatchClassification:
    """Test batch classification functionality."""
    
    def test_classify_batch(self, classifier):
        """Test classifying multiple texts at once."""
        texts = [
            "我住在北京",
            "我喜欢咖啡",
            "我是一个耐心的人",
        ]
        
        results = classifier.classify_batch(texts)
        
        assert len(results) == 3
        assert results[0].knowledge_type == KnowledgeType.FACTUAL_STATE
        assert results[1].knowledge_type == KnowledgeType.PREFERENCE
        assert results[2].knowledge_type == KnowledgeType.IDENTITY_TRAIT


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_text(self, classifier):
        """Test classification of empty text."""
        result = classifier.classify("")
        assert result.knowledge_type == KnowledgeType.UNKNOWN
        assert result.confidence < 0.5
    
    def test_whitespace_only(self, classifier):
        """Test classification of whitespace-only text."""
        result = classifier.classify("   \n\t  ")
        assert result.knowledge_type == KnowledgeType.UNKNOWN
    
    def test_mixed_language(self, classifier):
        """Test classification of mixed language text."""
        result = classifier.classify("我 live in 北京")
        assert result.knowledge_type == KnowledgeType.FACTUAL_STATE
    
    def test_long_text(self, classifier):
        """Test classification of long text."""
        long_text = "我住在北京，" * 100
        result = classifier.classify(long_text)
        assert result.knowledge_type == KnowledgeType.FACTUAL_STATE
    
    def test_special_characters(self, classifier):
        """Test classification with special characters."""
        result = classifier.classify("我住在北京！！！")
        assert result.knowledge_type == KnowledgeType.FACTUAL_STATE


# =============================================================================
# Test Reproducibility
# =============================================================================

class TestReproducibility:
    """Test that classification is deterministic with same seed."""
    
    def test_same_seed_same_results(self):
        """Test that same seed produces same results."""
        classifier1 = KnowledgeClassifier(seed=42)
        classifier2 = KnowledgeClassifier(seed=42)
        
        text = "我是一个耐心的人"
        
        result1 = classifier1.classify(text)
        result2 = classifier2.classify(text)
        
        assert result1.knowledge_type == result2.knowledge_type
        assert result1.confidence == result2.confidence
    
    def test_different_seed_same_deterministic_results(self):
        """Test that classification is deterministic regardless of seed.
        
        Note: The classifier uses deterministic pattern matching,
        so seed only affects potential stochastic components.
        """
        classifier1 = KnowledgeClassifier(seed=42)
        classifier2 = KnowledgeClassifier(seed=123)
        
        text = "我住在北京"
        
        result1 = classifier1.classify(text)
        result2 = classifier2.classify(text)
        
        # Pattern matching is deterministic
        assert result1.knowledge_type == result2.knowledge_type


# =============================================================================
# Test First Principles Philosophy
# =============================================================================

class TestFirstPrinciples:
    """Test that the classifier follows first principles from narrative psychology."""
    
    def test_not_all_contradictions_need_elimination(self, classifier):
        """Test the core principle: not all contradictions need elimination."""
        # These should be resolved differently
        
        # State change → UPDATE
        state_analysis = classifier.resolve_conflict(
            KnowledgeType.FACTUAL_STATE,
            KnowledgeType.FACTUAL_STATE,
            "sequential",
            "我住在北京",
            "我住在上海",
        )
        assert state_analysis.resolution == ConflictResolution.UPDATE
        
        # Trait "contradiction" → PRESERVE_BOTH (functional flexibility)
        trait_analysis = classifier.resolve_conflict(
            KnowledgeType.IDENTITY_TRAIT,
            KnowledgeType.IDENTITY_TRAIT,
            "concurrent",
            "我是一个耐心的人",
            "我是一个高效的人",
        )
        assert trait_analysis.resolution == ConflictResolution.PRESERVE_BOTH
        assert trait_analysis.is_complementary
    
    def test_healthy_identity_contains_tensions(self, classifier):
        """Test that complementary traits are recognized as healthy, not contradictory."""
        # In narrative psychology, healthy identity contains productive tensions
        
        complementary_pairs = [
            ("我是一个耐心的人", "我是一个高效的人"),
            ("我比较内向", "我善于社交"),
            ("我很理性", "我也很感性"),
            ("我做事严谨", "我也能灵活变通"),
        ]
        
        for text_a, text_b in complementary_pairs:
            assert classifier.are_complementary_traits(text_a, text_b), \
                f"Expected complementary: '{text_a}' and '{text_b}'"
    
    def test_true_contradictions_are_identified(self, classifier):
        """Test that true moral/logical contradictions are correctly identified."""
        contradictory_pairs = [
            ("我很诚实", "我喜欢说谎"),
            ("我乐于帮助别人", "我喜欢伤害别人"),
        ]
        
        for text_a, text_b in contradictory_pairs:
            assert not classifier.are_complementary_traits(text_a, text_b), \
                f"Expected contradictory: '{text_a}' and '{text_b}'"
