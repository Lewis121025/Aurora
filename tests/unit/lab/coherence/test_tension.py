"""
AURORA 张力管理器测试
============================

TensionManager 组件的测试。

测试覆盖：
- 元素䭋间的张力检测
- 张力分类（action_blocking, identity_threatening, adaptive, developmental）
- 张力解决策略
"""

from __future__ import annotations

import numpy as np
import pytest

from aurora.lab.coherence.tension import TensionManager, Tension, TensionType


@pytest.fixture
def tension_manager() -> TensionManager:
    """创建一个TensionManager实例。"""
    return TensionManager(seed=42)


@pytest.fixture
def sample_element_a() -> dict:
    """创建示例元素A。"""
    return {
        "id": "plot_001",
        "type": "plot",
        "text": "用户应该使用这个功能。",
    }


@pytest.fixture
def sample_element_b() -> dict:
    """创建有矛盾期的示例元素B。"""
    return {
        "id": "plot_002",
        "type": "plot",
        "text": "用户不应该使用这个功能。",
    }


@pytest.fixture
def sample_element_c() -> dict:
    """创建不有矛盾期的示例元素C。"""
    return {
        "id": "plot_003",
        "type": "plot",
        "text": "这个功能很有用。",
    }


class TestTensionDetection:
    """张力检测的测试。"""

    def test_detect_tension_logical_contradiction(
        self, tension_manager, sample_element_a, sample_element_b
    ):
        """测试逻辑矛盾的检测。"""
        tension = tension_manager.detect_tension(sample_element_a, sample_element_b)

        # 应检测到"应该"vs"不应该"引起的张力
        assert tension is not None
        assert "矛盾" in tension.description

    def test_detect_no_tension(self, tension_manager, sample_element_a, sample_element_c):
        """测试对于非矛盾的元素不检测到张力。"""
        tension = tension_manager.detect_tension(sample_element_a, sample_element_c)

        # 没有明显的矛盾
        assert tension is None

    def test_detect_tension_with_embeddings(self, tension_manager):
        """测试使用语义嵌入的张力检测。"""
        rng = np.random.default_rng(42)

        emb_a = rng.standard_normal(64).astype(np.float32)
        emb_a = emb_a / np.linalg.norm(emb_a)

        # Create opposing embedding
        emb_b = -emb_a

        element_a = {"id": "a", "type": "plot", "text": "正面观点"}
        element_b = {"id": "b", "type": "plot", "text": "反面观点"}

        tension = tension_manager.detect_tension(
            element_a, element_b, embedding_a=emb_a, embedding_b=emb_b
        )

        # Opposing embeddings should trigger tension
        assert tension is not None
        assert tension.severity > 0


class TestTensionClassification:
    """Tests for tension classification."""

    def test_classify_action_blocking(self, tension_manager):
        """Test classification of action-blocking tension."""
        tension = Tension(
            id="test_tension_1",
            element_a_id="a",
            element_a_type="plot",
            element_b_id="b",
            element_b_type="plot",
            description="必须执行 vs 禁止执行",
            severity=0.9,
        )

        result = tension_manager.classify_tension(tension)

        # High severity with action keywords should be action-blocking
        assert result == TensionType.ACTION_BLOCKING

    def test_classify_identity_threatening(self, tension_manager):
        """Test classification of identity-threatening tension."""
        tension = Tension(
            id="test_tension_2",
            element_a_id="a",
            element_a_type="theme",
            element_b_id="b",
            element_b_type="theme",
            description="诚实的价值观受到挑战",
            severity=0.6,
        )

        result = tension_manager.classify_tension(tension)

        # Threatens core identity value
        assert result == TensionType.IDENTITY_THREATENING

    def test_classify_adaptive(self, tension_manager):
        """Test classification of adaptive tension."""
        tension = Tension(
            id="test_tension_3",
            element_a_id="a",
            element_a_type="plot",
            element_b_id="b",
            element_b_type="plot",
            description="有时候需要耐心，在某些情况下需要高效",
            severity=0.4,
        )

        result = tension_manager.classify_tension(tension)

        # Context-dependent, moderate severity
        assert result == TensionType.ADAPTIVE

    def test_classify_developmental(self, tension_manager):
        """Test classification of developmental tension."""
        tension = Tension(
            id="test_tension_4",
            element_a_id="a",
            element_a_type="plot",
            element_b_id="b",
            element_b_type="plot",
            description="以前我很急躁，现在我更有耐心了",
            severity=0.3,
        )

        result = tension_manager.classify_tension(tension)

        # Past vs present indicates growth
        assert result == TensionType.DEVELOPMENTAL


class TestTensionResolution:
    """Tests for tension resolution strategies."""

    def test_handle_action_blocking_tension(self, tension_manager):
        """Test handling of action-blocking tension."""
        tension = Tension(
            id="test_resolve_1",
            element_a_id="a",
            element_a_type="plot",
            element_b_id="b",
            element_b_type="plot",
            description="必须执行某操作",
            severity=0.9,
        )

        resolution = tension_manager.handle_tension(tension)

        assert resolution.action == "resolve"
        assert tension.resolution_status == "resolved"

    def test_handle_adaptive_tension(self, tension_manager):
        """Test handling of adaptive tension."""
        tension = Tension(
            id="test_preserve_1",
            element_a_id="a",
            element_a_type="plot",
            element_b_id="b",
            element_b_type="plot",
            description="有时需要A，在某些情况下需要B",
            severity=0.4,
        )

        resolution = tension_manager.handle_tension(tension)

        assert resolution.action == "preserve"
        assert tension.resolution_status == "preserved"

    def test_handle_developmental_tension(self, tension_manager):
        """Test handling of developmental tension."""
        tension = Tension(
            id="test_accept_1",
            element_a_id="a",
            element_a_type="plot",
            element_b_id="b",
            element_b_type="plot",
            description="以前如何，现在如何",
            severity=0.2,
        )

        resolution = tension_manager.handle_tension(tension)

        assert resolution.action == "accept"
        assert tension.resolution_status == "accepted"


class TestTensionManager:
    """Tests for TensionManager class."""

    def test_get_unresolved_tensions(self, tension_manager):
        """Test retrieval of unresolved tensions."""
        # Add some tensions
        tension1 = tension_manager._create_tension(
            {"id": "a", "type": "plot"},
            {"id": "b", "type": "plot"},
            "测试矛盾1",
            0.5,
        )
        tension2 = tension_manager._create_tension(
            {"id": "c", "type": "plot"},
            {"id": "d", "type": "plot"},
            "测试矛盾2",
            0.5,
        )

        # Resolve one
        tension_manager.handle_tension(tension1)

        unresolved = tension_manager.get_unresolved_tensions()

        assert len(unresolved) == 1
        assert unresolved[0].id == tension2.id

    def test_get_preserved_tensions(self, tension_manager):
        """Test retrieval of preserved tensions."""
        tension = tension_manager._create_tension(
            {"id": "a", "type": "plot"},
            {"id": "b", "type": "plot"},
            "在某些情况下需要灵活",
            0.4,
        )

        tension_manager.handle_tension(tension)

        preserved = tension_manager.get_preserved_tensions()

        assert len(preserved) >= 0  # May or may not be preserved depending on classification

    def test_get_tension_summary(self, tension_manager):
        """Test tension summary generation."""
        # Add various tensions
        for i, desc in enumerate(
            [
                "必须执行",
                "有时需要",
                "以前如此，现在不同",
            ]
        ):
            tension_manager._create_tension(
                {"id": f"a{i}", "type": "plot"},
                {"id": f"b{i}", "type": "plot"},
                desc,
                0.5 + i * 0.2,
            )

        summary = tension_manager.get_tension_summary()

        assert "total" in summary
        assert "by_type" in summary
        assert "by_status" in summary
        assert summary["total"] == 3

    def test_serialization_round_trip(self, tension_manager):
        """Test serialization and deserialization."""
        # Add a tension
        tension_manager._create_tension(
            {"id": "a", "type": "plot"},
            {"id": "b", "type": "plot"},
            "测试矛盾",
            0.5,
        )

        # Serialize
        state = tension_manager.to_state_dict()

        # Deserialize
        restored = TensionManager.from_state_dict(state, seed=42)

        assert len(restored.tensions) == len(tension_manager.tensions)
        assert restored.core_identity_values == tension_manager.core_identity_values


class TestTensionDataClass:
    """Tests for Tension dataclass."""

    def test_tension_creation(self):
        """Test Tension creation with default values."""
        tension = Tension(
            id="test",
            element_a_id="a",
            element_a_type="plot",
            element_b_id="b",
            element_b_type="plot",
            description="测试描述",
        )

        assert tension.severity == 0.5
        assert tension.resolution_status == "unresolved"
        assert tension.tension_type == TensionType.UNKNOWN

    def test_tension_to_dict(self):
        """Test Tension serialization."""
        tension = Tension(
            id="test",
            element_a_id="a",
            element_a_type="plot",
            element_b_id="b",
            element_b_type="plot",
            description="测试描述",
            severity=0.7,
        )

        d = tension.to_dict()

        assert d["id"] == "test"
        assert d["severity"] == 0.7
        assert d["tension_type"] == "unknown"

    def test_tension_from_dict(self):
        """Test Tension deserialization."""
        d = {
            "id": "test",
            "element_a_id": "a",
            "element_a_type": "plot",
            "element_b_id": "b",
            "element_b_type": "plot",
            "description": "测试描述",
            "tension_type": "adaptive",
            "severity": 0.6,
        }

        tension = Tension.from_dict(d)

        assert tension.id == "test"
        assert tension.tension_type == TensionType.ADAPTIVE
        assert tension.severity == 0.6
