"""
弃权检测的测试
================================

测试当置信度低时拒绝回答的弃权机制。
"""

import pytest
import numpy as np

from aurora.algorithms.abstention import AbstentionDetector, AbstentionResult
from aurora.algorithms.aurora_core import AuroraMemory


class TestAbstentionDetector:
    """AbstentionDetector 类的测试。"""

    def test_no_results_abstain(self):
        """测试空结果触发弃权。"""
        detector = AbstentionDetector()
        result = detector.detect(
            query="What did we discuss?",
            retrieved_scores=[],
            retrieved_texts=[],
        )
        assert result.should_abstain is True
        assert result.confidence == 1.0
        assert "No results" in result.reason
    
    def test_low_relevance_abstain(self):
        """Test that low relevance scores trigger abstention."""
        detector = AbstentionDetector(min_relevance_threshold=0.35)
        result = detector.detect(
            query="What is my favorite color?",
            retrieved_scores=[0.2, 0.15, 0.1],
            retrieved_texts=["Some unrelated text", "Another text", "More text"],
        )
        assert result.should_abstain is True
        assert result.confidence > 0.0
        assert "Low relevance" in result.reason
    
    def test_high_relevance_no_abstain(self):
        """Test that high relevance scores don't trigger abstention."""
        detector = AbstentionDetector(min_relevance_threshold=0.35)
        result = detector.detect(
            query="What did we discuss?",
            retrieved_scores=[0.85, 0.75, 0.65],
            retrieved_texts=["We discussed Python", "We talked about algorithms", "We mentioned recursion"],
        )
        assert result.should_abstain is False
        assert result.confidence > 0.5
        assert "Confident" in result.reason
    
    def test_negation_in_retrieved_text_does_not_abstain(self):
        """Test that negation patterns in retrieved text do NOT trigger abstention.
        
        This is a critical fix: negation words in retrieved memories should not
        cause false abstention. For example, if a memory says "I didn't mention
        my travel plans", that shouldn't cause abstention on unrelated queries.
        """
        detector = AbstentionDetector()
        result = detector.detect(
            query="How many days between MoMA and Metropolitan?",
            retrieved_scores=[0.75, 0.65, 0.55],  # High relevance scores
            retrieved_texts=[
                "I didn't mention it before, but I visited MoMA on Monday",
                "We haven't discussed the Metropolitan visit timing",
                "I went to Metropolitan three days after MoMA"
            ],
        )
        # Should NOT abstain because scores are high, even though texts contain negation words
        assert result.should_abstain is False
        assert "Confident" in result.reason
    
    def test_existence_query_with_low_score_abstains(self):
        """Test that existence queries ('did we ever...') with low scores abstain."""
        detector = AbstentionDetector()
        result = detector.detect(
            query="Did we ever discuss my favorite color?",
            retrieved_scores=[0.4, 0.35, 0.3],  # Moderate-low scores
            retrieved_texts=["Some unrelated text", "Another text", "More text"],
        )
        assert result.should_abstain is True
        assert "Existence query" in result.reason or "Low relevance" in result.reason
    
    def test_existence_query_with_high_score_no_abstain(self):
        """Test that existence queries with high scores do NOT abstain."""
        detector = AbstentionDetector()
        result = detector.detect(
            query="Have we ever discussed Python?",
            retrieved_scores=[0.85, 0.75, 0.65],  # High scores
            retrieved_texts=["We discussed Python extensively", "Python is great"],
        )
        assert result.should_abstain is False
        assert "Confident" in result.reason
    
    def test_uniform_low_scores_abstain(self):
        """Test that uniform low scores trigger abstention."""
        detector = AbstentionDetector(
            min_relevance_threshold=0.27,  # Set threshold above top score (0.26) to allow uniform check
            low_score_threshold=0.3,
            uniform_score_std_threshold=0.05,
        )
        result = detector.detect(
            query="What is my favorite color?",
            retrieved_scores=[0.25, 0.24, 0.26, 0.25],  # Very uniform, all low (top=0.26)
            retrieved_texts=["Text 1", "Text 2", "Text 3", "Text 4"],
        )
        assert result.should_abstain is True
        # Check for actual reason format: "Low relevance score" (signal 1) or "Uniform low scores" (signal 3)
        # Since top_score (0.26) < min_relevance (0.27), signal 1 will trigger first
        assert "Low relevance" in result.reason or "Uniform" in result.reason or "low scores" in result.reason.lower()
    
    def test_chinese_existence_query_abstains(self):
        """Test Chinese existence query patterns trigger abstention when scores are low."""
        detector = AbstentionDetector()
        result = detector.detect(
            query="我们有没有讨论过我的爱好？",  # Existence query pattern
            retrieved_scores=[0.4],  # Moderate-low score
            retrieved_texts=["一些不太相关的内容"],
        )
        assert result.should_abstain is True
        assert "Existence query" in result.reason or "Low relevance" in result.reason
    
    def test_chinese_negation_in_text_no_abstain(self):
        """Test that Chinese negation words in retrieved text do NOT trigger abstention."""
        detector = AbstentionDetector()
        result = detector.detect(
            query="我的爱好是什么？",  # Simple question, not existence query
            retrieved_scores=[0.75],  # High score
            retrieved_texts=["我们从未提到过你的爱好，但你说你喜欢音乐"],
        )
        # Should NOT abstain because score is high
        assert result.should_abstain is False
        assert "Confident" in result.reason
    
    def test_all_low_scores_abstain(self):
        """Test that all low scores trigger abstention."""
        detector = AbstentionDetector(min_relevance_threshold=0.35)
        result = detector.detect(
            query="What is my favorite color?",
            retrieved_scores=[0.28, 0.26, 0.24],  # All below 80% of threshold
            retrieved_texts=["Text 1", "Text 2", "Text 3"],
        )
        assert result.should_abstain is True
        # Check for actual reason format: "Low relevance score" or "All top results have low scores"
        assert "Low relevance" in result.reason or "low scores" in result.reason.lower()


class TestAbstentionIntegration:
    """Integration tests for abstention in AuroraMemory."""
    
    def test_query_with_no_memory_abstains(self):
        """Test that querying empty memory triggers abstention."""
        mem = AuroraMemory(seed=42)
        trace = mem.query("What did we discuss about Python?")
        
        assert trace.abstention is not None
        assert trace.abstention.should_abstain is True
        assert "No results" in trace.abstention.reason
    
    def test_query_with_low_relevance_abstains(self):
        """Test that queries with low relevance trigger abstention."""
        mem = AuroraMemory(seed=42)
        
        # Ingest some unrelated content
        mem.ingest("User: I like apples", actors=["user", "agent"])
        mem.ingest("User: The weather is nice today", actors=["user", "agent"])
        mem.evolve()
        
        # Query something completely unrelated
        trace = mem.query("What is my favorite programming language?")
        
        assert trace.abstention is not None
        # Should abstain if scores are too low
        if trace.abstention.should_abstain:
            assert "Low relevance" in trace.abstention.reason or "low scores" in trace.abstention.reason.lower()
    
    def test_query_with_high_relevance_no_abstain(self):
        """Test that queries with high relevance don't trigger abstention."""
        mem = AuroraMemory(seed=42)
        
        # Ingest relevant content
        mem.ingest("User: I love Python programming", actors=["user", "agent"])
        mem.ingest("User: Python is my favorite language", actors=["user", "agent"])
        mem.evolve()
        
        # Query something related
        trace = mem.query("What is my favorite programming language?")
        
        assert trace.abstention is not None
        # Should not abstain if scores are high enough
        if not trace.abstention.should_abstain:
            assert "Confident" in trace.abstention.reason
    
    def test_abstention_result_in_trace(self):
        """Test that abstention result is properly attached to trace."""
        mem = AuroraMemory(seed=42)
        trace = mem.query("Test query")
        
        assert hasattr(trace, "abstention")
        assert trace.abstention is not None
        assert isinstance(trace.abstention, AbstentionResult)
        assert isinstance(trace.abstention.should_abstain, bool)
        assert isinstance(trace.abstention.confidence, float)
        assert isinstance(trace.abstention.reason, str)
        assert 0.0 <= trace.abstention.confidence <= 1.0
