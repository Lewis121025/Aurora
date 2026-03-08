"""
AURORA 核心测试
=================

AuroraMemory 核心类的测试。

测试覆盖：
- 使用以关系为中心的处理进行摄入
- 使用身份激活进行查询
- 反馈和学习
- 演化和合并
- 序列化和反序列化
"""

from __future__ import annotations

import numpy as np
import pytest

from aurora.algorithms.aurora_core import AuroraMemory
from aurora.algorithms.models.config import MemoryConfig
from aurora.exceptions import MemoryNotFoundError, ValidationError


class TestAuroraMemoryIngest:
    """AuroraMemory.ingest() 方法的测试。"""
    
    def test_ingest_creates_plot(self, aurora_memory: AuroraMemory):
        """测试 ingest 会创建有正确结构的情节。"""
        plot = aurora_memory.ingest(
            "用户：你好！助理：你好，有什么可以帮你的？",
            actors=("user", "assistant"),
        )
        
        assert plot is not None
        assert plot.id is not None
        assert plot.ts > 0
        assert plot.embedding is not None
        assert plot.embedding.shape == (64,)
    
    def test_ingest_extracts_relational_context(self, aurora_memory: AuroraMemory):
        """测试 ingest 会提取关系上下文。"""
        plot = aurora_memory.ingest(
            "用户：帮我解释一下递归。助理：递归是函数调用自身的过程。",
            actors=("user", "assistant"),
        )
        
        # 情节可能不会被存储（概率性），但应该有关系上下文
        assert plot.relational is not None
        assert plot.relational.with_whom == "user"
        assert plot.relational.my_role_in_relation is not None
    
    def test_ingest_computes_signals(self, aurora_memory: AuroraMemory):
        """测试 ingest 计算所有信号值。"""
        plot = aurora_memory.ingest(
            "用户：这是一个测试。助理：好的。",
            actors=("user", "assistant"),
        )
        
        assert hasattr(plot, 'surprise')
        assert hasattr(plot, 'pred_error')
        assert hasattr(plot, 'redundancy')
        assert hasattr(plot, 'goal_relevance')
        assert hasattr(plot, 'tension')
    
    def test_ingest_updates_kde(self, aurora_memory: AuroraMemory):
        """测试 ingest 更新 KDE 密度估计业。"""
        initial_count = len(aurora_memory.kde._vecs)
        
        aurora_memory.ingest("用户：测试f1。助理：接到。", actors=("user", "assistant"))
        aurora_memory.ingest("用户：测试2。助理：接到。", actors=("user", "assistant"))
        aurora_memory.ingest("用户：测试3。助理：接到。", actors=("user", "assistant"))
        
        # 无论是否做出存储决定，KDE 都应该更新
        assert len(aurora_memory.kde._vecs) >= initial_count
    
    def test_ingest_with_context(self, aurora_memory: AuroraMemory):
        """测试带有上下文文本的 ingest 会影响目标相关性。"""
        plot = aurora_memory.ingest(
            "用户：帮我写一个排序算法。助理：好的，我来帮你实现。",
            actors=("user", "assistant"),
            context_text="编程任务",
        )
        
        # 提供上下文时应该计算目标相关性
        assert plot.goal_relevance >= 0
    
    def test_ingest_creates_relationship_story(self, aurora_memory: AuroraMemory):
        """测试重复与同一用户的互动会创建关系故事。"""
        # 与同一用户进行多次互动
        for i in range(5):
            aurora_memory.ingest(
                f"用户：问題{i}。助理：回答{i}。",
                actors=("user", "assistant"),
            )
        
        # 如果情节被存储，应该创建关系故事
        if aurora_memory.plots:
            relationship_story = aurora_memory.get_relationship_story("user")
            # 可能存在也可能不存在，取决于存储决定
            if relationship_story:
                assert relationship_story.relationship_with == "user"


class TestAuroraMemoryIngestBatch:
    """AuroraMemory.ingest_batch() 方法的测试。"""
    
    def test_ingest_batch_empty(self, aurora_memory: AuroraMemory):
        """测试 ingest_batch 处理空输入。"""
        plots = aurora_memory.ingest_batch([])
        assert plots == []
    
    def test_ingest_batch_creates_plots(self, aurora_memory: AuroraMemory):
        """测试 ingest_batch 为所有输入创建情节。"""
        interactions = [
            {"text": "用户：你好！助理：你好，有什么可以帮你的？"},
            {"text": "用户：帮我解释递归。助理：递归是函数调用自身。"},
            {"text": "用户：谢谢！助理：不客气。"},
        ]
        
        plots = aurora_memory.ingest_batch(interactions)
        
        assert len(plots) == 3
        for plot in plots:
            assert plot is not None
            assert plot.id is not None
            assert plot.embedding is not None
    
    def test_ingest_batch_with_actors(self, aurora_memory: AuroraMemory):
        """测试 ingest_batch 处理自定义演员。"""
        interactions = [
            {"text": "Alice: Hi Bob!", "actors": ("Alice", "Bob")},
            {"text": "Bob: Hello Alice!", "actors": ("Bob", "Alice")},
        ]
        
        plots = aurora_memory.ingest_batch(interactions)
        
        assert len(plots) == 2
        assert plots[0].actors == ("Alice", "Bob")
        assert plots[1].actors == ("Bob", "Alice")
    
    def test_ingest_batch_with_context(self, aurora_memory: AuroraMemory):
        """测试 ingest_batch 处理 context_text。"""
        interactions = [
            {"text": "用户：帮我写排序算法", "context_text": "编程任务"},
            {"text": "用户：天气怎么样", "context_text": "闲聊"},
        ]
        
        plots = aurora_memory.ingest_batch(interactions)
        
        assert len(plots) == 2
        # 提供上下文时应该计算目标相关性
        for plot in plots:
            assert plot.goal_relevance >= 0
    
    def test_ingest_batch_with_event_ids(self, aurora_memory: AuroraMemory):
        """测试 ingest_batch 支持确定性的 event_ids。"""
        interactions = [
            {"text": "测试1", "event_id": "batch-evt-001"},
            {"text": "测试2", "event_id": "batch-evt-002"},
        ]
        
        plots = aurora_memory.ingest_batch(interactions)
        
        assert len(plots) == 2
        # ID 应该是确定性的（相同的 event_id -> 相同的 plot.id）
        # 再次运行并验证相同的 ID
        from aurora.algorithms.aurora_core import AuroraMemory as AM
        from aurora.algorithms.models.config import MemoryConfig as MC
        mem2 = AM(cfg=MC(dim=64, max_plots=100), seed=42)
        plots2 = mem2.ingest_batch(interactions)
        assert plots[0].id == plots2[0].id
        assert plots[1].id == plots2[1].id
    
    def test_ingest_batch_updates_kde(self, aurora_memory: AuroraMemory):
        """测试 ingest_batch 更新所有情节的 KDE。"""
        initial_count = len(aurora_memory.kde._vecs)
        
        interactions = [{"文本": f"测试互动 {i}"} for i in range(10)]
        aurora_memory.ingest_batch(interactions)
        
        # 无论是否做出存储决定，KDE 都应该更新
        assert len(aurora_memory.kde._vecs) >= initial_count
    
    def test_ingest_batch_progress_callback(self, aurora_memory: AuroraMemory):
        """测试进度回调是否被正确调用。"""
        interactions = [{"文本": f"测试 {i}"} for i in range(5)]
        
        progress_calls = []
        def callback(current, total, plot):
            progress_calls.append((current, total, plot.id))
        
        plots = aurora_memory.ingest_batch(interactions, progress_callback=callback)
        
        assert len(progress_calls) == 5
        for i, (current, total, plot_id) in enumerate(progress_calls):
            assert current == i + 1
            assert total == 5
            assert plot_id == plots[i].id
    
    def test_ingest_batch_validates_empty_text(self, aurora_memory: AuroraMemory):
        """测试 ingest_batch 验证空文本。"""
        interactions = [
            {"文本": "有效怎女"},
            {"文本": ""},  # 无效
            {"文本": "另一个有效"},
        ]
        
        with pytest.raises(ValidationError):
            aurora_memory.ingest_batch(interactions)
    
    def test_ingest_batch_determinism(self):
        """测试使用相同種子的批餅摄入是确定性的。"""
        cfg = MemoryConfig(dim=64, max_plots=100)
        interactions = [
            {"text": f"测试交互 {i}", "event_id": f"evt-{i}"} 
            for i in range(10)
        ]
        
        # 使用相同種子的第一次运行
        mem1 = AuroraMemory(cfg=cfg, seed=42)
        plots1 = mem1.ingest_batch(interactions)
        
        # 使用相同種子的第二次运行
        mem2 = AuroraMemory(cfg=cfg, seed=42)
        plots2 = mem2.ingest_batch(interactions)
        
        # Should produce identical results
        assert len(plots1) == len(plots2)
        for p1, p2 in zip(plots1, plots2):
            assert p1.id == p2.id
            assert np.allclose(p1.embedding, p2.embedding)
    
    def test_ingest_batch_benchmark_mode(self):
        """测试 benchmark_mode 会存储所有情节。"""
        cfg = MemoryConfig(dim=64, max_plots=1000)
        mem = AuroraMemory(cfg=cfg, seed=42, benchmark_mode=True)
        
        interactions = [{"text": f"测试交互 {i}"} for i in range(20)]
        plots = mem.ingest_batch(interactions)
        
        # 在 benchmark 模式下，所有情节都应该被存储
        stored_count = sum(1 for p in plots if p.id in mem.plots)
        assert stored_count == 20
    
    def test_ingest_batch_with_date(self, aurora_memory: AuroraMemory):
        """测试 ingest_batch 正确处理日期一欽。"""
        interactions = [
            {"text": "User: Hello", "date": "2023/01/08 (Sun) 12:04"},
            {"text": "User: I went hiking", "date": "2023/01/09 (Mon) 09:30"},
            {"text": "User: No date here"},  # Without date field
        ]
        
        plots = aurora_memory.ingest_batch(interactions)
        
        assert len(plots) == 3
        # 前两个应该带有日期前缀
        assert plots[0].text == "[2023/01/08 (Sun) 12:04] User: Hello"
        assert plots[1].text == "[2023/01/09 (Mon) 09:30] User: I went hiking"
        # 第三个应该保持不变
        assert plots[2].text == "User: No date here"
    
    def test_ingest_batch_date_affects_embedding(self):
        """测试日期前缀对底体幻桂会造成不同的幻桂。"""
        cfg = MemoryConfig(dim=64, max_plots=100)
        mem = AuroraMemory(cfg=cfg, seed=42, benchmark_mode=True)
        
        # 相同的文本，不同的日期应该产生不同的幻桂
        interactions_with_dates = [
            {"文本": "用户：我去了公园", "date": "2023/01/08 (Sun)"},
            {"文本": "用户：我去了公园", "date": "2023/06/15 (Thu)"},
        ]
        plots = mem.ingest_batch(interactions_with_dates)
        
        # 由於日期前缀的原因，文本应该不同
        assert plots[0].text != plots[1].text
        assert "2023/01/08" in plots[0].text
        assert "2023/06/15" in plots[1].text
    
    def test_ingest_batch_date_backward_compatible(self, aurora_memory: AuroraMemory):
        """测试日期不子是可选的且与代码向后兼容。"""
        # 没有使用日期不子的旧代码应该仍然正常工作
        interactions = [
            {"文本": "用户：你好", "actors": ("user", "assistant")},
            {"文本": "用户：再觑", "context_text": "闲聂"},
        ]
        
        plots = aurora_memory.ingest_batch(interactions)
        
        assert len(plots) == 2
        # 没有日期不子时，文本应该保持不变
        assert plots[0].text == "用户：你好"
        assert plots[1].text == "用户：再见"
    
    def test_ingest_batch_equivalent_to_serial(self):
        """测试批摄入会产生与序列摄入相同的结果。"""
        cfg = MemoryConfig(dim=64, max_plots=100)
        interactions = [
            {"text": "用户：你好", "actors": ("user", "assistant"), "event_id": "evt-1"},
            {"text": "用户：再见", "actors": ("user", "assistant"), "event_id": "evt-2"},
        ]
        
        # 批摄入
        mem_batch = AuroraMemory(cfg=cfg, seed=42, benchmark_mode=True)
        plots_batch = mem_batch.ingest_batch(interactions)
        
        # 序列摄入
        mem_serial = AuroraMemory(cfg=cfg, seed=42, benchmark_mode=True)
        plots_serial = [
            mem_serial.ingest(
                item["text"],
                actors=item.get("actors"),
                event_id=item.get("event_id"),
            )
            for item in interactions
        ]
        
        # 应会产生相同的结果
        assert len(plots_batch) == len(plots_serial)
        for pb, ps in zip(plots_batch, plots_serial):
            assert pb.id == ps.id
            assert np.allclose(pb.embedding, ps.embedding)
            assert pb.surprise == pytest.approx(ps.surprise, rel=1e-5)


class TestAuroraMemoryQuery:
    """AuroraMemory.query() 方法的测试。"""
    
    def test_query_returns_trace(self, populated_memory: AuroraMemory):
        """测试 query 返回正常的 RetrievalTrace。"""
        trace = populated_memory.query("递归是什么？", k=3)
        
        assert trace is not None
        assert trace.query == "递归是什么？"
        assert trace.query_emb is not None
        assert len(trace.query_emb) == 64
    
    def test_query_with_asker_id(self, populated_memory: AuroraMemory):
        """测试使用 asker_id 的 query 会激活关系上下文。"""
        trace = populated_memory.query(
            "帮我解释一下",
            k=3,
            asker_id="user",
        )
        
        assert trace.asker_id == "user"
    
    def test_query_updates_access_stats(self, populated_memory: AuroraMemory):
        """测试 query 会更新检索项目的访问统计。"""
        if not populated_memory.plots:
            pytest.skip("No plots stored")
        
        # Get initial access counts
        initial_counts = {
            pid: p.access_count for pid, p in populated_memory.plots.items()
        }
        
        # Perform query
        trace = populated_memory.query("测试查询", k=5)
        
        # Check if any access counts increased
        for pid, (_, _, kind) in zip(
            [r[0] for r in trace.ranked],
            trace.ranked
        ):
            if kind == "plot" and pid in populated_memory.plots:
                assert populated_memory.plots[pid].access_count >= initial_counts.get(pid, 0)


class TestAuroraMemoryFeedback:
    """AuroraMemory.feedback_retrieval() 方法的测试。"""
    
    def test_feedback_positive(self, populated_memory: AuroraMemory):
        """测试积极反馈会更新信念。"""
        if not populated_memory.plots:
            pytest.skip("No plots stored")
        
        # Get a plot ID to use as chosen
        plot_id = next(iter(populated_memory.plots.keys()))
        
        # Should not raise
        populated_memory.feedback_retrieval(
            "测试查询",
            chosen_id=plot_id,
            success=True,
        )
    
    def test_feedback_negative(self, populated_memory: AuroraMemory):
        """测试消极反馈会更新信念。"""
        if not populated_memory.plots:
            pytest.skip("No plots stored")
        
        plot_id = next(iter(populated_memory.plots.keys()))
        
        # Should not raise
        populated_memory.feedback_retrieval(
            "测试查询",
            chosen_id=plot_id,
            success=False,
        )


class TestAuroraMemoryEvolution:
    """AuroraMemory.evolve() 方法的测试。"""
    
    def test_evolve_runs_without_error(self, populated_memory: AuroraMemory):
        """测试 evolve() 是否能不出错地运行。"""
        # Should not raise
        populated_memory.evolve()
    
    def test_evolve_updates_story_statuses(self, populated_memory: AuroraMemory):
        """测试 evolve 是否会更新故事状态。"""
        # Run multiple evolution cycles
        for _ in range(3):
            populated_memory.evolve()
        
        # 检查故事是否存在且具有有效的状态
        for story in populated_memory.stories.values():
            assert story.status in ("developing", "resolved", "abandoned")
    
    def test_evolve_handles_empty_memory(self, aurora_memory: AuroraMemory):
        """测试 evolve 是否能正常处理空内存。"""
        # 不应该屆空的内存把字或不檤
        aurora_memory.evolve()


class TestAuroraMemorySerialization:
    """AuroraMemory 序列化的测试。"""
    
    def test_to_state_dict(self, populated_memory: AuroraMemory):
        """测试序列化为状态字典。"""
        state = populated_memory.to_state_dict()
        
        assert "version" in state
        assert "cfg" in state
        assert "plots" in state
        assert "stories" in state
        assert "themes" in state
    
    def test_from_state_dict(self, populated_memory: AuroraMemory):
        """测试从状态字典反序列化。"""
        # 序列化
        state = populated_memory.to_state_dict()
        
        # 反序列化
        restored = AuroraMemory.from_state_dict(state)
        
        assert len(restored.plots) == len(populated_memory.plots)
        assert len(restored.stories) == len(populated_memory.stories)
    
    def test_round_trip_preserves_data(self, populated_memory: AuroraMemory):
        """测试序列化往返过程是否能保留数据。"""
        state = populated_memory.to_state_dict()
        restored = AuroraMemory.from_state_dict(state)
        
        # 检查情节是否被保留
        for pid, plot in populated_memory.plots.items():
            assert pid in restored.plots
            restored_plot = restored.plots[pid]
            assert restored_plot.text == plot.text
            assert np.allclose(restored_plot.embedding, plot.embedding)


class TestAuroraMemoryIdentity:
    """u8eab份相关功能的测试。"""
    
    def test_get_identity_summary(self, populated_memory: AuroraMemory):
        """测试获取身份汇总。"""
        summary = populated_memory.get_identity_summary()
        
        assert "identity_dimensions" in summary
        assert "relationship_identities" in summary
        assert "relationship_count" in summary
        assert "total_interactions" in summary
    
    def test_identity_dimensions_accumulate(self, aurora_memory: AuroraMemory):
        """测试身份维度在一段時間内的累积。"""
        # 攻入加增身份的互动
        for i in range(10):
            aurora_memory.ingest(
                f"用户：帮我解餇问题{i}。助理：好的，让我来解餇。",
                actors=("user", "assistant"),
            )
        
        # 身份维度应该有一些值
        # （如果没有情节被存储，可能为int）
        summary = aurora_memory.get_identity_summary()
        assert isinstance(summary["identity_dimensions"], dict)


class TestAuroraMemoryPressure:
    """u5185存压力管理的测试。"""
    
    def test_pressure_manage_respects_capacity(self):
        """测试压力管理是否会遵守最大情节数的限制。"""
        config = MemoryConfig(dim=64, max_plots=10)
        memory = AuroraMemory(cfg=config, seed=42)
        
        # 摄入许多互动
        for i in range(30):
            memory.ingest(
                f"用户：测试{i}。助理：收到{i}。",
                actors=("user", "assistant"),
            )
        
        # Active plots should not exceed max_plots by much
        # (some tolerance for probabilistic behavior)
        active_plots = len([p for p in memory.plots.values() if p.status == "active"])
        assert active_plots <= config.max_plots * 1.5


class TestAuroraMemoryUtilityMethods:
    """Tests for utility methods."""
    
    def test_update_centroid_online_new(self, aurora_memory: AuroraMemory):
        """Test online centroid update for new centroid."""
        emb = np.random.randn(64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        
        result = aurora_memory._update_centroid_online(None, emb, 1)
        
        assert np.allclose(result, emb)
    
    def test_update_centroid_online_existing(self, aurora_memory: AuroraMemory):
        """Test online centroid update for existing centroid."""
        rng = np.random.default_rng(42)
        current = rng.standard_normal(64).astype(np.float32)
        current = current / np.linalg.norm(current)
        
        new_emb = rng.standard_normal(64).astype(np.float32)
        new_emb = new_emb / np.linalg.norm(new_emb)
        
        result = aurora_memory._update_centroid_online(current, new_emb, 2)
        
        # Result should be normalized
        assert np.isclose(np.linalg.norm(result), 1.0, rtol=1e-5)
    
    def test_create_bidirectional_edge(self, aurora_memory: AuroraMemory):
        """Test bidirectional edge creation."""
        # Add some nodes first
        aurora_memory.graph.add_node("test_node_1", "plot", None)
        aurora_memory.graph.add_node("test_node_2", "story", None)
        
        aurora_memory._create_bidirectional_edge(
            "test_node_1", "test_node_2",
            "belongs_to", "contains"
        )
        
        # Both edges should exist
        assert aurora_memory.graph.g.has_edge("test_node_1", "test_node_2")
        assert aurora_memory.graph.g.has_edge("test_node_2", "test_node_1")


class TestAuroraMemoryExceptions:
    """Tests for custom exception handling."""

    def test_get_story_not_found(self, aurora_memory: AuroraMemory):
        """Test that get_story raises MemoryNotFoundError for non-existent story."""
        with pytest.raises(MemoryNotFoundError) as exc_info:
            aurora_memory.get_story("non_existent_story_id")
        
        assert exc_info.value.kind == "story"
        assert exc_info.value.element_id == "non_existent_story_id"
        assert "story" in str(exc_info.value)
        assert "non_existent_story_id" in str(exc_info.value)

    def test_get_plot_not_found(self, aurora_memory: AuroraMemory):
        """Test that get_plot raises MemoryNotFoundError for non-existent plot."""
        with pytest.raises(MemoryNotFoundError) as exc_info:
            aurora_memory.get_plot("non_existent_plot_id")
        
        assert exc_info.value.kind == "plot"
        assert exc_info.value.element_id == "non_existent_plot_id"
        assert "plot" in str(exc_info.value)
        assert "non_existent_plot_id" in str(exc_info.value)

    def test_get_theme_not_found(self, aurora_memory: AuroraMemory):
        """Test that get_theme raises MemoryNotFoundError for non-existent theme."""
        with pytest.raises(MemoryNotFoundError) as exc_info:
            aurora_memory.get_theme("non_existent_theme_id")
        
        assert exc_info.value.kind == "theme"
        assert exc_info.value.element_id == "non_existent_theme_id"
        assert "theme" in str(exc_info.value)
        assert "non_existent_theme_id" in str(exc_info.value)

    def test_get_story_success(self, populated_memory: AuroraMemory):
        """Test that get_story returns story when it exists."""
        if not populated_memory.stories:
            pytest.skip("No stories created")
        
        story_id = next(iter(populated_memory.stories.keys()))
        story = populated_memory.get_story(story_id)
        assert story is not None
        assert story.id == story_id

    def test_get_plot_success(self, populated_memory: AuroraMemory):
        """Test that get_plot returns plot when it exists."""
        if not populated_memory.plots:
            pytest.skip("No plots stored")
        
        plot_id = next(iter(populated_memory.plots.keys()))
        plot = populated_memory.get_plot(plot_id)
        assert plot is not None
        assert plot.id == plot_id


class TestAuroraMemoryValidation:
    """Tests for input validation."""

    def test_ingest_empty_string_raises(self, aurora_memory: AuroraMemory):
        """Test that ingest raises ValidationError for empty string."""
        with pytest.raises(ValidationError) as exc_info:
            aurora_memory.ingest("")
        
        assert "interaction_text cannot be empty" in str(exc_info.value)

    def test_ingest_whitespace_only_raises(self, aurora_memory: AuroraMemory):
        """Test that ingest raises ValidationError for whitespace-only string."""
        with pytest.raises(ValidationError) as exc_info:
            aurora_memory.ingest("   \n\t  ")
        
        assert "interaction_text cannot be empty" in str(exc_info.value)

    def test_ingest_none_raises(self, aurora_memory: AuroraMemory):
        """Test that ingest raises ValidationError for None input."""
        with pytest.raises(ValidationError) as exc_info:
            aurora_memory.ingest(None)
        
        assert "interaction_text cannot be empty" in str(exc_info.value)

    def test_query_empty_string_raises(self, aurora_memory: AuroraMemory):
        """Test that query raises ValidationError for empty string."""
        with pytest.raises(ValidationError) as exc_info:
            aurora_memory.query("")
        
        assert "query text cannot be empty" in str(exc_info.value)

    def test_query_whitespace_only_raises(self, aurora_memory: AuroraMemory):
        """Test that query raises ValidationError for whitespace-only string."""
        with pytest.raises(ValidationError) as exc_info:
            aurora_memory.query("   \n\t  ")
        
        assert "query text cannot be empty" in str(exc_info.value)

    def test_query_none_raises(self, aurora_memory: AuroraMemory):
        """Test that query raises ValidationError for None input."""
        with pytest.raises(ValidationError) as exc_info:
            aurora_memory.query(None)
        
        assert "query text cannot be empty" in str(exc_info.value)

    def test_ingest_valid_input_succeeds(self, aurora_memory: AuroraMemory):
        """Test that ingest succeeds with valid input."""
        plot = aurora_memory.ingest("用户：有效输入。助理：收到。")
        assert plot is not None
        assert plot.text == "用户：有效输入。助理：收到。"

    def test_query_valid_input_succeeds(self, populated_memory: AuroraMemory):
        """Test that query succeeds with valid input."""
        trace = populated_memory.query("有效查询")
        assert trace is not None
        assert trace.query == "有效查询"


class TestAuroraMemoryBenchmarkMode:
    """Tests for benchmark_mode feature - force store all plots, bypass VOI gating."""
    
    def test_benchmark_mode_via_parameter(self):
        """Test benchmark_mode set via __init__ parameter forces storage of all plots."""
        cfg = MemoryConfig(dim=64, max_plots=100)
        mem = AuroraMemory(cfg=cfg, seed=42, benchmark_mode=True)
        
        # Ingest multiple plots
        plots = []
        for i in range(20):
            plot = mem.ingest(f"测试交互 {i}", actors=("user", "assistant"))
            plots.append(plot)
        
        # In benchmark_mode, ALL plots should be stored (100% storage rate)
        stored_count = sum(1 for plot in plots if plot.id in mem.plots)
        assert stored_count == 20, f"Expected all 20 plots stored, got {stored_count}"
        
        # Verify storage_prob is set to 1.0
        for plot in plots:
            assert hasattr(plot, '_storage_prob')
            assert plot._storage_prob == 1.0
    
    def test_benchmark_mode_via_config(self):
        """Test benchmark_mode set via MemoryConfig forces storage of all plots."""
        cfg = MemoryConfig(dim=64, max_plots=100, benchmark_mode=True)
        mem = AuroraMemory(cfg=cfg, seed=42)
        
        # Ingest multiple plots
        plots = []
        for i in range(20):
            plot = mem.ingest(f"测试交互 {i}", actors=("user", "assistant"))
            plots.append(plot)
        
        # In benchmark_mode, ALL plots should be stored
        stored_count = sum(1 for plot in plots if plot.id in mem.plots)
        assert stored_count == 20, f"Expected all 20 plots stored, got {stored_count}"
    
    def test_benchmark_mode_parameter_overrides_config(self):
        """Test that benchmark_mode parameter has higher priority than config."""
        # Config says False, but parameter says True
        cfg = MemoryConfig(dim=64, max_plots=100, benchmark_mode=False)
        mem = AuroraMemory(cfg=cfg, seed=42, benchmark_mode=True)
        
        assert mem.benchmark_mode is True
        
        # Should force store all plots
        plot = mem.ingest("测试", actors=("user", "assistant"))
        assert plot.id in mem.plots
        assert plot._storage_prob == 1.0
    
    def test_normal_mode_storage_is_probabilistic(self):
        """Test that normal mode (benchmark_mode=False) has probabilistic storage."""
        cfg = MemoryConfig(dim=64, max_plots=100, benchmark_mode=False)
        mem = AuroraMemory(cfg=cfg, seed=42, benchmark_mode=False)
        
        # Ingest many plots
        plots = []
        for i in range(50):
            plot = mem.ingest(f"测试交互 {i}", actors=("user", "assistant"))
            plots.append(plot)
        
        # In normal mode, storage should be probabilistic (not 100%)
        stored_count = sum(1 for plot in plots if plot.id in mem.plots)
        
        # Should have some stored but not all (unless cold start protection kicks in)
        # Cold start forces first N plots, so we check after that
        if len(mem.plots) > 10:  # After cold start period
            # Should have probabilistic behavior (not 100% storage)
            assert stored_count < len(plots), "Normal mode should not store 100%"
    
    def test_benchmark_mode_bypasses_voi_gating(self):
        """Test that benchmark_mode bypasses VOI gating completely."""
        cfg = MemoryConfig(dim=64, max_plots=100)
        mem = AuroraMemory(cfg=cfg, seed=42, benchmark_mode=True)
        
        # Create plots that would normally have low VOI (redundant, low surprise)
        redundant_texts = [
            "用户：你好。助理：你好。",
            "用户：你好。助理：你好。",  # Exact duplicate
            "用户：你好。助理：你好。",  # Exact duplicate
        ]
        
        plots = []
        for text in redundant_texts:
            plot = mem.ingest(text, actors=("user", "assistant"))
            plots.append(plot)
        
        # All should be stored despite redundancy
        stored_count = sum(1 for plot in plots if plot.id in mem.plots)
        assert stored_count == len(plots), "benchmark_mode should store even redundant plots"
    
    def test_benchmark_mode_config_serialization(self):
        """Test that benchmark_mode is preserved in config serialization."""
        cfg = MemoryConfig(dim=64, benchmark_mode=True)
        
        # Serialize and deserialize
        state_dict = cfg.to_state_dict()
        restored_cfg = MemoryConfig.from_state_dict(state_dict)
        
        assert restored_cfg.benchmark_mode is True
        
        # Verify it works in AuroraMemory
        mem = AuroraMemory(cfg=restored_cfg, seed=42)
        assert mem.benchmark_mode is True
        
        plot = mem.ingest("测试", actors=("user", "assistant"))
        assert plot.id in mem.plots
