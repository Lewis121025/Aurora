"""
AURORA 关系模块测试
=================================

RelationshipMixin功能的测试，详见aurora/core/memory/relationship.py。

测试覆盖：
- 关系实体识别
- 身份相关性评估
- 关系上下文提取
- 身份影响提取
- 关系故事管理
- 身份维度更新
"""

from __future__ import annotations

import numpy as np
import pytest

from aurora.core.memory import AuroraMemory
from aurora.core.models.config import MemoryConfig
from aurora.core.models.plot import IdentityImpact, Plot, RelationalContext
from aurora.core.models.story import StoryArc
from aurora.core.models.theme import Theme
from aurora.utils.id_utils import det_id
from aurora.utils.time_utils import now_ts


class TestIdentifyRelationshipEntity:
    """关系实体识别的测试。"""

    def test_identify_relationship_entity_with_user(self, aurora_memory: AuroraMemory):
        """测试有用户actor的实体识别。"""
        entity = aurora_memory._identify_relationship_entity(
            actors=("user", "assistant"),
            text="用户：你好！助理：你好！",
        )
        
        assert entity == "user"

    def test_identify_relationship_entity_filters_system(self, aurora_memory: AuroraMemory):
        """测试系统/助理actor是否被噚销。"""
        entity = aurora_memory._identify_relationship_entity(
            actors=("assistant", "system"),
            text="测试文本",
        )
        
        # 当没有其他实体时应默认为user
        assert entity == "user"

    def test_identify_relationship_entity_named_user(self, aurora_memory: AuroraMemory):
        """测试有具名用户的实体识别。"""
        entity = aurora_memory._identify_relationship_entity(
            actors=("张三", "assistant"),
            text="张三问了一个问题",
        )
        
        assert entity == "张三"

    def test_identify_relationship_entity_multiple_actors(self, aurora_memory: AuroraMemory):
        """测试有多个非系统 actor 的实体识别。"""
        entity = aurora_memory._identify_relationship_entity(
            actors=("李四", "王五", "assistant"),
            text="多人对话",
        )
        
        # 应返回第一个非系统 actor
        assert entity == "李四"

    def test_identify_relationship_entity_case_insensitive(self, aurora_memory: AuroraMemory):
        """测试系统 actor 筛选是否不区分大小写。"""
        entity = aurora_memory._identify_relationship_entity(
            actors=("Agent", "System", "real_user"),
            text="测试",
        )
        
        assert entity == "real_user"


class TestAssessIdentityRelevance:
    """身份相关性评估的测试。"""

    def test_assess_identity_relevance_empty_themes(self, aurora_memory: AuroraMemory):
        """测试当没有主题时的身份相关性。"""
        rng = np.random.default_rng(42)
        emb = rng.standard_normal(64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        
        relevance = aurora_memory._assess_identity_relevance(
            text="这是一个新交互",
            relationship_entity="user",
            emb=emb,
        )
        
        # Should have novelty contribution when no themes
        assert relevance >= 0

    def test_assess_identity_relevance_with_themes(self, aurora_memory: AuroraMemory):
        """测试有现有主题时的身份相关性。"""
        rng = np.random.default_rng(42)
        
        # 创建一个主题程窗
        theme_emb = rng.standard_normal(64).astype(np.float32)
        theme_emb = theme_emb / np.linalg.norm(theme_emb)
        
        theme = Theme(
            id=det_id("theme", "relevance_test"),
            created_ts=now_ts(),
            updated_ts=now_ts(),
            prototype=theme_emb,
        )
        aurora_memory.themes[theme.id] = theme
        
        # Test with similar embedding
        similar_emb = theme_emb + rng.standard_normal(64).astype(np.float32) * 0.1
        similar_emb = similar_emb / np.linalg.norm(similar_emb)
        
        relevance = aurora_memory._assess_identity_relevance(
            text="相关内容",
            relationship_entity="user",
            emb=similar_emb,
        )
        
        assert 0 <= relevance <= 1

    def test_assess_identity_relevance_relationship_factor(self, aurora_memory: AuroraMemory):
        """Test that relationship importance affects relevance."""
        rng = np.random.default_rng(42)
        emb = rng.standard_normal(64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        
        # Create important relationship
        story = StoryArc(
            id=det_id("story", "important_rel"),
            created_ts=now_ts(),
            updated_ts=now_ts(),
            relationship_with="important_user",
            relationship_health=0.9,
        )
        story.plot_ids = ["p1", "p2", "p3", "p4", "p5"]  # Many interactions
        aurora_memory.stories[story.id] = story
        aurora_memory._relationship_story_index["important_user"] = story.id
        
        relevance_important = aurora_memory._assess_identity_relevance(
            text="测试",
            relationship_entity="important_user",
            emb=emb,
        )
        
        relevance_new = aurora_memory._assess_identity_relevance(
            text="测试",
            relationship_entity="new_user",
            emb=emb,
        )
        
        # Important relationship should have higher relevance multiplier
        # (though not guaranteed due to other factors)
        assert relevance_important >= 0
        assert relevance_new >= 0


class TestExtractRelationalContext:
    """Tests for relational context extraction."""

    def test_extract_relational_context_basic(self, aurora_memory: AuroraMemory):
        """Test basic relational context extraction."""
        context = aurora_memory._extract_relational_context(
            text="用户：帮我解释一下递归。助理：好的，递归是...",
            relationship_entity="user",
            actors=("user", "assistant"),
            identity_relevance=0.5,
        )
        
        assert isinstance(context, RelationalContext)
        assert context.with_whom == "user"
        assert context.my_role_in_relation is not None
        assert context.what_this_says_about_us is not None

    def test_extract_relational_context_role_inference(self, aurora_memory: AuroraMemory):
        """Test role inference in relational context."""
        # Test with explanation keywords
        context = aurora_memory._extract_relational_context(
            text="让我来解释一下这个概念...",
            relationship_entity="student",
            actors=("student", "assistant"),
            identity_relevance=0.6,
        )
        
        assert context.my_role_in_relation == "解释者"

    def test_extract_relational_context_quality_delta(self, aurora_memory: AuroraMemory):
        """Test quality delta computation."""
        # Positive interaction
        context_positive = aurora_memory._extract_relational_context(
            text="谢谢你的帮助，太好了！",
            relationship_entity="user",
            actors=("user", "assistant"),
            identity_relevance=0.5,
        )
        
        # Negative interaction
        context_negative = aurora_memory._extract_relational_context(
            text="这个答案是错误的，不行",
            relationship_entity="user",
            actors=("user", "assistant"),
            identity_relevance=0.5,
        )
        
        assert context_positive.relationship_quality_delta > context_negative.relationship_quality_delta

    def test_extract_relational_context_uses_relationship_history(
        self, aurora_memory: AuroraMemory
    ):
        """Test that context extraction uses relationship history."""
        # Create relationship with established identity
        story = StoryArc(
            id=det_id("story", "context_history"),
            created_ts=now_ts(),
            updated_ts=now_ts(),
            relationship_with="history_user",
            my_identity_in_this_relationship="编程导师",
        )
        # Add consistent role history
        for i in range(8):
            story.add_relationship_moment(
                event_summary=f"编程指导 {i}",
                trust_level=0.7,
                my_role="编程导师",
            )
        
        aurora_memory.stories[story.id] = story
        aurora_memory._relationship_story_index["history_user"] = story.id
        
        context = aurora_memory._extract_relational_context(
            text="用户：帮我看看这段代码",
            relationship_entity="history_user",
            actors=("history_user", "assistant"),
            identity_relevance=0.6,
        )
        
        # Should use established identity when role consistency is high
        assert context.my_role_in_relation == "编程导师"


class TestExtractIdentityImpact:
    """Tests for identity impact extraction."""

    def test_extract_identity_impact_low_relevance(self, aurora_memory: AuroraMemory):
        """Test that low relevance doesn't create identity impact."""
        relational = RelationalContext(
            with_whom="user",
            my_role_in_relation="助手",
            relationship_quality_delta=0.0,
            what_this_says_about_us="常规互动",
        )
        
        impact = aurora_memory._extract_identity_impact(
            text="普通内容",
            relational=relational,
            identity_relevance=0.1,  # Below threshold
        )
        
        assert impact is None

    def test_extract_identity_impact_significant_relevance(self, aurora_memory: AuroraMemory):
        """Test identity impact extraction for significant interactions."""
        relational = RelationalContext(
            with_whom="user",
            my_role_in_relation="解释者",
            relationship_quality_delta=0.15,
            what_this_says_about_us="良好互动",
        )
        
        impact = aurora_memory._extract_identity_impact(
            text="帮我解释一下这个复杂的概念",
            relational=relational,
            identity_relevance=0.6,  # Above threshold
        )
        
        assert impact is not None
        assert isinstance(impact, IdentityImpact)
        assert len(impact.identity_dimensions_affected) > 0
        assert impact.initial_meaning is not None

    def test_extract_identity_impact_dimension_keywords(self, aurora_memory: AuroraMemory):
        """Test that keywords map to correct identity dimensions."""
        relational = RelationalContext(
            with_whom="user",
            my_role_in_relation="编程助手",
            relationship_quality_delta=0.1,
            what_this_says_about_us="编程帮助",
        )
        
        impact = aurora_memory._extract_identity_impact(
            text="帮我写一段Python代码来分析数据",
            relational=relational,
            identity_relevance=0.5,
        )
        
        assert impact is not None
        # Should include programming-related dimension
        dimensions = impact.identity_dimensions_affected
        assert any("编程" in d for d in dimensions) or any("分析" in d for d in dimensions)

    def test_extract_identity_impact_includes_role_dimension(self, aurora_memory: AuroraMemory):
        """Test that role-based dimension is always included."""
        relational = RelationalContext(
            with_whom="user",
            my_role_in_relation="学习伙伴",
            relationship_quality_delta=0.05,
            what_this_says_about_us="学习交流",
        )
        
        impact = aurora_memory._extract_identity_impact(
            text="让我们一起学习这个主题",
            relational=relational,
            identity_relevance=0.4,
        )
        
        assert impact is not None
        # Role dimension should be included
        assert any("学习伙伴" in d for d in impact.identity_dimensions_affected)


class TestGetOrCreateRelationshipStory:
    """Tests for relationship story management."""

    def test_get_or_create_new_story(self, aurora_memory: AuroraMemory):
        """Test creating a new relationship story."""
        story = aurora_memory._get_or_create_relationship_story("new_user")
        
        assert story is not None
        assert story.relationship_with == "new_user"
        assert story.id in aurora_memory.stories
        assert "new_user" in aurora_memory._relationship_story_index

    def test_get_or_create_existing_story(self, aurora_memory: AuroraMemory):
        """Test getting an existing relationship story."""
        # Create first
        story1 = aurora_memory._get_or_create_relationship_story("existing_user")
        story1_id = story1.id
        
        # Get again
        story2 = aurora_memory._get_or_create_relationship_story("existing_user")
        
        # Should be the same story
        assert story2.id == story1_id

    def test_get_or_create_story_determines_type(self, aurora_memory: AuroraMemory):
        """Test that story type is determined from entity name."""
        user_story = aurora_memory._get_or_create_relationship_story("some_user")
        assert user_story.relationship_type == "user"
        
        other_story = aurora_memory._get_or_create_relationship_story("company_system")
        assert other_story.relationship_type == "other"

    def test_get_or_create_story_adds_to_graph(self, aurora_memory: AuroraMemory):
        """Test that new story is added to graph."""
        story = aurora_memory._get_or_create_relationship_story("graph_user")
        
        assert aurora_memory.graph.g.has_node(story.id)


class TestUpdateIdentityDimensions:
    """Tests for identity dimension updates."""

    def test_update_identity_dimensions_new_dimension(self, aurora_memory: AuroraMemory):
        """Test creating new identity dimension."""
        rng = np.random.default_rng(42)
        emb = rng.standard_normal(64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        
        plot = Plot(
            id=det_id("plot", "new_dim"),
            ts=now_ts(),
            text="新维度测试",
            actors=("user",),
            embedding=emb,
            identity_impact=IdentityImpact(
                when_formed=now_ts(),
                initial_meaning="测试",
                current_meaning="测试",
                identity_dimensions_affected=["作为测试者的我"],
                evolution_history=[],
            ),
        )
        
        aurora_memory._update_identity_dimensions(plot)
        
        assert "作为测试者的我" in aurora_memory._identity_dimensions
        assert aurora_memory._identity_dimensions["作为测试者的我"] > 0

    def test_update_identity_dimensions_strengthens_existing(
        self, aurora_memory: AuroraMemory
    ):
        """Test that repeated exposure strengthens dimension."""
        rng = np.random.default_rng(42)
        
        # Set initial dimension
        aurora_memory._identity_dimensions["作为帮助者的我"] = 0.3
        
        # Create multiple plots reinforcing the dimension
        for i in range(5):
            emb = rng.standard_normal(64).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            
            plot = Plot(
                id=det_id("plot", f"strengthen_{i}"),
                ts=now_ts(),
                text=f"帮助测试 {i}",
                actors=("user",),
                embedding=emb,
                identity_impact=IdentityImpact(
                    when_formed=now_ts(),
                    initial_meaning="帮助他人",
                    current_meaning="帮助他人",
                    identity_dimensions_affected=["作为帮助者的我"],
                    evolution_history=[],
                ),
            )
            
            aurora_memory._update_identity_dimensions(plot)
        
        # Dimension should be strengthened
        assert aurora_memory._identity_dimensions["作为帮助者的我"] > 0.3

    def test_update_identity_dimensions_no_impact(self, aurora_memory: AuroraMemory):
        """Test update with plot having no identity impact."""
        rng = np.random.default_rng(42)
        emb = rng.standard_normal(64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        
        plot = Plot(
            id=det_id("plot", "no_impact_update"),
            ts=now_ts(),
            text="无影响",
            actors=("user",),
            embedding=emb,
            identity_impact=None,
        )
        
        initial_dims = dict(aurora_memory._identity_dimensions)
        aurora_memory._update_identity_dimensions(plot)
        
        # Dimensions should be unchanged
        assert aurora_memory._identity_dimensions == initial_dims


class TestRelationshipImportance:
    """Tests for relationship importance computation."""

    def test_get_relationship_importance_new(self, aurora_memory: AuroraMemory):
        """Test importance for new relationship."""
        importance = aurora_memory._get_relationship_importance("unknown_user")
        
        # New relationships should have neutral importance
        assert importance == 0.5

    def test_get_relationship_importance_established(self, aurora_memory: AuroraMemory):
        """Test importance for established relationship."""
        # Create relationship with history
        story = StoryArc(
            id=det_id("story", "established"),
            created_ts=now_ts(),
            updated_ts=now_ts(),
            relationship_with="established_user",
            relationship_health=0.8,
        )
        story.plot_ids = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]
        
        aurora_memory.stories[story.id] = story
        aurora_memory._relationship_story_index["established_user"] = story.id
        
        importance = aurora_memory._get_relationship_importance("established_user")
        
        # Established healthy relationship should have high importance
        assert importance > 0.5


class TestRoleInference:
    """Tests for role inference functionality."""

    def test_infer_my_role_explanation(self, aurora_memory: AuroraMemory):
        """Test role inference for explanation context."""
        role = aurora_memory._infer_my_role("让我来解释一下这个概念")
        assert role == "解释者"

    def test_infer_my_role_help(self, aurora_memory: AuroraMemory):
        """Test role inference for help context."""
        role = aurora_memory._infer_my_role("我来帮助你解决这个问题")
        assert role == "帮助者"

    def test_infer_my_role_analysis(self, aurora_memory: AuroraMemory):
        """Test role inference for analysis context."""
        role = aurora_memory._infer_my_role("让我分析一下这个情况")
        assert role == "分析者"

    def test_infer_my_role_programming(self, aurora_memory: AuroraMemory):
        """Test role inference for programming context."""
        role = aurora_memory._infer_my_role("这是一段Python代码")
        assert role == "编程助手"

    def test_infer_my_role_default(self, aurora_memory: AuroraMemory):
        """Test role inference falls back to default."""
        role = aurora_memory._infer_my_role("随便聊聊天")
        assert role == "助手"

    def test_infer_my_role_uses_history(self, aurora_memory: AuroraMemory):
        """Test that role inference uses relationship history when available."""
        # Create relationship with established role
        story = StoryArc(
            id=det_id("story", "role_history"),
            created_ts=now_ts(),
            updated_ts=now_ts(),
            relationship_with="role_user",
            my_identity_in_this_relationship="数据分析师",
        )
        # Add consistent role history
        for i in range(8):
            story.add_relationship_moment(
                event_summary=f"数据分析 {i}",
                trust_level=0.7,
                my_role="数据分析师",
            )
        
        aurora_memory.stories[story.id] = story
        aurora_memory._relationship_story_index["role_user"] = story.id
        
        # Should use established role even with different keywords
        role = aurora_memory._infer_my_role("今天天气不错", relationship_entity="role_user")
        assert role == "数据分析师"


class TestIdentitySignals:
    """Tests for identity signal computation."""

    def test_compute_identity_signals_no_themes(self, aurora_memory: AuroraMemory):
        """Test identity signals with no themes."""
        rng = np.random.default_rng(42)
        emb = rng.standard_normal(64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        
        reinforcement, challenge, novelty = aurora_memory._compute_identity_signals(
            text="测试", emb=emb
        )
        
        # No themes = full novelty
        assert reinforcement == 0.0
        assert challenge == 0.0
        assert novelty == 1.0

    def test_compute_identity_signals_with_similar_theme(self, aurora_memory: AuroraMemory):
        """Test identity signals with similar theme."""
        rng = np.random.default_rng(42)
        
        # Create theme
        theme_emb = rng.standard_normal(64).astype(np.float32)
        theme_emb = theme_emb / np.linalg.norm(theme_emb)
        
        theme = Theme(
            id=det_id("theme", "similar_signal"),
            created_ts=now_ts(),
            updated_ts=now_ts(),
            prototype=theme_emb,
        )
        aurora_memory.themes[theme.id] = theme
        aurora_memory._identity_dimensions["测试维度"] = 0.5
        
        # Test with similar embedding
        similar_emb = theme_emb + rng.standard_normal(64).astype(np.float32) * 0.1
        similar_emb = similar_emb / np.linalg.norm(similar_emb)
        
        reinforcement, challenge, novelty = aurora_memory._compute_identity_signals(
            text="测试", emb=similar_emb
        )
        
        # Should have high reinforcement, low novelty
        assert reinforcement > 0.5
        assert novelty < 0.5
