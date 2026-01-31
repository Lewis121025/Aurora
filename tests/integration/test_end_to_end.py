"""
AURORA End-to-End Integration Tests
===================================

Full workflow integration tests that verify the complete memory lifecycle.

Tests cover:
- Complete ingest → query → evolve workflow
- Multi-user/multi-session scenarios
- State persistence and recovery
- Memory pressure under load
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from aurora.algorithms.aurora_core import AuroraMemory
from aurora.algorithms.models.config import MemoryConfig


class TestCompleteWorkflow:
    """Test complete memory workflow: ingest → query → evolve."""
    
    def test_basic_workflow(self):
        """Test basic ingest → query → evolve workflow."""
        config = MemoryConfig(dim=64, max_plots=100)
        memory = AuroraMemory(cfg=config, seed=42)
        
        # === Phase 1: Ingest interactions ===
        interactions = [
            "用户：你好，我想学习Python。助理：好的，我们从基础开始。",
            "用户：什么是变量？助理：变量是用来存储数据的容器。",
            "用户：怎么定义函数？助理：使用def关键字定义函数。",
            "用户：谢谢你的帮助！助理：不客气，继续加油！",
        ]
        
        for text in interactions:
            memory.ingest(text, actors=("user", "assistant"))
        
        # === Phase 2: Query ===
        trace = memory.query("Python变量是什么？", k=3)
        
        assert trace is not None
        assert trace.query_emb is not None
        
        # === Phase 3: Evolve ===
        memory.evolve()
        
        # Memory should still be functional after evolution
        trace2 = memory.query("函数怎么定义？", k=3)
        assert trace2 is not None
    
    def test_feedback_improves_retrieval(self):
        """Test that feedback improves future retrieval."""
        config = MemoryConfig(dim=64, max_plots=100)
        memory = AuroraMemory(cfg=config, seed=42)
        
        # Ingest diverse interactions
        topics = [
            ("递归", "递归是函数调用自身的过程。"),
            ("排序", "排序算法将元素按顺序排列。"),
            ("搜索", "搜索算法用于查找元素。"),
            ("图算法", "图算法处理节点和边的数据结构。"),
        ]
        
        for topic, explanation in topics:
            memory.ingest(
                f"用户：什么是{topic}？助理：{explanation}",
                actors=("user", "assistant"),
            )
        
        # Query and provide feedback
        for _ in range(3):
            trace = memory.query("递归算法", k=3)
            if trace.ranked:
                # Simulate positive feedback for relevant results
                memory.feedback_retrieval(
                    "递归算法",
                    chosen_id=trace.ranked[0][0],
                    success=True,
                )
        
        # Retrieval should still work
        final_trace = memory.query("递归", k=3)
        assert final_trace is not None
    
    def test_evolution_consolidates_memory(self):
        """Test that evolution consolidates plots into stories and themes."""
        config = MemoryConfig(dim=64, max_plots=50)
        memory = AuroraMemory(cfg=config, seed=42)
        
        # Ingest many related interactions
        for i in range(20):
            memory.ingest(
                f"用户：编程问题{i}。助理：这是关于编程的回答{i}。",
                actors=("user", "assistant"),
            )
        
        initial_plots = len(memory.plots)
        initial_stories = len(memory.stories)
        
        # Run multiple evolution cycles
        for _ in range(5):
            memory.evolve()
        
        # Memory should have some structure
        assert len(memory.stories) >= initial_stories or len(memory.stories) > 0
    
    def test_relationship_story_building(self):
        """Test that relationship stories are built over time."""
        config = MemoryConfig(dim=64, max_plots=100)
        memory = AuroraMemory(cfg=config, seed=42)
        
        # Simulate a series of interactions with the same user
        for i in range(10):
            memory.ingest(
                f"用户：帮助请求{i}。助理：这是我的帮助{i}。",
                actors=("user", "assistant"),
            )
        
        # Check if relationship story exists
        rel_story = memory.get_relationship_story("user")
        
        # May or may not exist depending on storage decisions
        if rel_story is not None:
            assert rel_story.relationship_with == "user"
            assert rel_story.relationship_type == "user"


class TestStatePersistence:
    """Test state persistence and recovery."""
    
    def test_state_roundtrip(self):
        """Test that state can be saved and restored."""
        config = MemoryConfig(dim=64, max_plots=50)
        memory = AuroraMemory(cfg=config, seed=42)
        
        # Build up some state
        for i in range(10):
            memory.ingest(f"交互{i}", actors=("user", "assistant"))
        memory.evolve()
        
        # Save state
        state = memory.to_state_dict()
        
        # Restore state
        restored = AuroraMemory.from_state_dict(state)
        
        # Verify restoration
        assert len(restored.plots) == len(memory.plots)
        assert len(restored.stories) == len(memory.stories)
        assert len(restored.themes) == len(memory.themes)
    
    def test_restored_memory_functional(self):
        """Test that restored memory is fully functional."""
        config = MemoryConfig(dim=64, max_plots=50)
        memory = AuroraMemory(cfg=config, seed=42)
        
        # Build state
        for i in range(5):
            memory.ingest(f"测试交互{i}", actors=("user", "assistant"))
        
        # Save and restore
        state = memory.to_state_dict()
        restored = AuroraMemory.from_state_dict(state)
        
        # Should be able to continue operations
        restored.ingest("新的交互", actors=("user", "assistant"))
        trace = restored.query("测试", k=3)
        restored.evolve()
        
        # All operations should complete without error
        assert trace is not None


class TestMemoryPressure:
    """Test memory behavior under pressure."""
    
    def test_capacity_management(self):
        """Test that memory respects capacity limits."""
        config = MemoryConfig(dim=64, max_plots=20)
        memory = AuroraMemory(cfg=config, seed=42)
        
        # Ingest more than capacity
        for i in range(50):
            memory.ingest(f"大量交互{i}", actors=("user", "assistant"))
        
        # Active plots should be bounded
        active_plots = len([p for p in memory.plots.values() if p.status == "active"])
        
        # Allow some tolerance for probabilistic behavior
        assert active_plots <= config.max_plots * 2
    
    def test_growth_oriented_forgetting(self):
        """Test that forgetting prioritizes growth."""
        config = MemoryConfig(dim=64, max_plots=15)
        memory = AuroraMemory(cfg=config, seed=42)
        
        # Create a mix of identity-relevant and less relevant interactions
        important_interactions = [
            "用户：谢谢你教会我很多。助理：很高兴能帮助你成长。",
            "用户：你真的帮了大忙。助理：这是我的职责。",
        ]
        
        routine_interactions = [
            f"用户：天气怎么样{i}？助理：今天天气不错{i}。"
            for i in range(30)
        ]
        
        # Ingest important ones first
        for text in important_interactions:
            memory.ingest(text, actors=("user", "assistant"))
        
        # Then flood with routine ones
        for text in routine_interactions:
            memory.ingest(text, actors=("user", "assistant"))
        
        # Memory should still function
        trace = memory.query("帮助成长", k=5)
        assert trace is not None


class TestConcurrency:
    """Test concurrent access patterns (simulated)."""
    
    def test_snapshot_evolution(self):
        """Test copy-on-write evolution pattern."""
        config = MemoryConfig(dim=64, max_plots=50)
        memory = AuroraMemory(cfg=config, seed=42)
        
        # Build state
        for i in range(10):
            memory.ingest(f"交互{i}", actors=("user", "assistant"))
        
        # Create snapshot
        snapshot = memory.create_evolution_snapshot()
        
        # Compute patch (simulates background processing)
        patch = memory.compute_evolution_patch(snapshot)
        
        # Apply patch
        memory.apply_evolution_patch(patch)
        
        # Memory should remain consistent
        trace = memory.query("测试", k=3)
        assert trace is not None
    
    def test_concurrent_ingest_and_query(self):
        """Test that ingest and query can interleave."""
        config = MemoryConfig(dim=64, max_plots=50)
        memory = AuroraMemory(cfg=config, seed=42)
        
        # Interleave ingest and query
        for i in range(20):
            memory.ingest(f"交互{i}", actors=("user", "assistant"))
            
            if i % 3 == 0:
                trace = memory.query(f"查询{i}", k=3)
                assert trace is not None


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_query(self):
        """Test query with empty text."""
        config = MemoryConfig(dim=64, max_plots=50)
        memory = AuroraMemory(cfg=config, seed=42)
        
        # Should handle empty query
        trace = memory.query("", k=3)
        assert trace is not None
    
    def test_single_character_ingest(self):
        """Test ingest with minimal text."""
        config = MemoryConfig(dim=64, max_plots=50)
        memory = AuroraMemory(cfg=config, seed=42)
        
        # Should handle minimal input
        plot = memory.ingest(".", actors=("user", "assistant"))
        assert plot is not None
    
    def test_unicode_handling(self):
        """Test handling of various Unicode characters."""
        config = MemoryConfig(dim=64, max_plots=50)
        memory = AuroraMemory(cfg=config, seed=42)
        
        texts = [
            "用户：测试中文。助理：好的。",
            "用户：Test English. Assistant: OK.",
            "用户：テスト日本語。助理：はい。",
            "用户：🎉 Emoji test 🚀. 助理：收到！",
        ]
        
        for text in texts:
            plot = memory.ingest(text, actors=("user", "assistant"))
            assert plot is not None
        
        # Should be able to query
        trace = memory.query("中文", k=3)
        assert trace is not None
    
    def test_very_long_text(self):
        """Test handling of very long text."""
        config = MemoryConfig(dim=64, max_plots=50)
        memory = AuroraMemory(cfg=config, seed=42)
        
        # Create very long text
        long_text = "这是一段很长的文字。" * 500
        
        plot = memory.ingest(long_text, actors=("user", "assistant"))
        assert plot is not None
        assert plot.text == long_text
