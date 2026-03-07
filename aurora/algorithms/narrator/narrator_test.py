"""叙述者模块的测试。

此模块测试叙事重构引擎及其组件：
- NarratorEngine: 主要叙事重构引擎
- PerspectiveSelector: 贝叶斯视角选择
- PerspectiveOrganizer: 特定视角的情节组织
- ContextRecovery: 因果上下文恢复和转折点检测
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest

from aurora.algorithms.components.metric import LowRankMetric
from aurora.algorithms.models.plot import IdentityImpact, Plot, RelationalContext
from aurora.algorithms.narrator import (
    ContextRecovery,
    NarrativeElement,
    NarrativePerspective,
    NarrativeRole,
    NarrativeTrace,
    NarratorEngine,
    PerspectiveOrganizer,
    PerspectiveSelector,
    TurningPointDetector,
)
from aurora.utils.id_utils import det_id
from aurora.utils.time_utils import now_ts


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def narrator_metric() -> LowRankMetric:
    """为叙述者测试创建LowRankMetric。"""
    return LowRankMetric(dim=64, rank=16, seed=42)


@pytest.fixture
def rng() -> np.random.Generator:
    """创建具有固定种子的随机生成器。"""
    return np.random.default_rng(42)


@pytest.fixture
def narrator_engine(narrator_metric: LowRankMetric) -> NarratorEngine:
    """创建NarratorEngine实例。"""
    return NarratorEngine(metric=narrator_metric, seed=42)


@pytest.fixture
def perspective_selector(
    narrator_metric: LowRankMetric, rng: np.random.Generator
) -> PerspectiveSelector:
    """创建PerspectiveSelector实例。"""
    return PerspectiveSelector(metric=narrator_metric, rng=rng)


@pytest.fixture
def perspective_organizer(
    narrator_metric: LowRankMetric, rng: np.random.Generator
) -> PerspectiveOrganizer:
    """创建PerspectiveOrganizer实例。"""
    return PerspectiveOrganizer(metric=narrator_metric, rng=rng)


@pytest.fixture
def context_recovery(
    narrator_metric: LowRankMetric, rng: np.random.Generator
) -> ContextRecovery:
    """创建ContextRecovery实例。"""
    return ContextRecovery(metric=narrator_metric, rng=rng)


@pytest.fixture
def turning_point_detector(rng: np.random.Generator) -> TurningPointDetector:
    """创建TurningPointDetector实例。"""
    return TurningPointDetector(rng=rng)


def make_plot(
    idx: int,
    rng: np.random.Generator,
    ts_offset: float = 0.0,
    tension: float = 0.5,
    text: str = "",
    story_id: str | None = None,
    with_identity_impact: bool = False,
) -> Plot:
    """创建测试Plot的辅助函数。

    注意：ts_offset应为负数或零以确保情节在过去。
    """
    emb = rng.standard_normal(64).astype(np.float32)
    emb = emb / np.linalg.norm(emb)

    identity_impact = None
    if with_identity_impact:
        identity_impact = IdentityImpact(
            when_formed=now_ts(),
            initial_meaning="测试身份影响",
            current_meaning="测试身份影响",
            identity_dimensions_affected=["作为助手的我"],
            evolution_history=[],
        )

    # 使用负偏移以确保情节在过去
    # 这防止了log1p(now - plot.ts)中的数学域错误
    base_ts = now_ts() - 86400  # 从1天前开始
    return Plot(
        id=det_id("plot", f"test_{idx}"),
        ts=base_ts + ts_offset,
        text=text or f"测试交互 {idx}",
        actors=("user", "assistant"),
        embedding=emb,
        tension=tension,
        story_id=story_id,
        identity_impact=identity_impact,
    )


@pytest.fixture
def sample_plots_for_narrator(rng: np.random.Generator) -> List[Plot]:
    """创建具有不同张力和时间戳的测试情节列表。"""
    plots = []
    texts = [
        "用户：今天天气不错。助理：是的，很适合出去走走。",
        "用户：我遇到了一个困难的问题。助理：让我来帮你分析一下。",
        "用户：问题终于解决了！助理：太好了，坚持就有收获。",
        "用户：我想学习新技能。助理：这是很好的想法。",
        "用户：谢谢你一直的帮助。助理：很高兴能帮到你。",
    ]
    tensions = [0.3, 0.8, 0.4, 0.6, 0.2]

    for i, (text, tension) in enumerate(zip(texts, tensions)):
        plot = make_plot(
            idx=i,
            rng=rng,
            ts_offset=i * 3600,  # 相隔1小时（相对于make_plot中的base_ts）
            tension=tension,
            text=text,
            with_identity_impact=(i == 2),  # 第三个情节有身份影响
        )
        plots.append(plot)

    return plots


# =============================================================================
# NarratorEngine Tests
# =============================================================================


class TestNarratorEngine:
    """NarratorEngine类的测试。"""

    def test_reconstruct_story_empty_input(self, narrator_engine: NarratorEngine):
        """使用空情节列表测试reconstruct_story。"""
        trace = narrator_engine.reconstruct_story(
            query="测试查询",
            plots=[],
        )

        assert isinstance(trace, NarrativeTrace)
        assert trace.query == "测试查询"
        assert trace.elements == []
        assert trace.perspective == NarrativePerspective.CHRONOLOGICAL
        assert "没有找到" in trace.narrative_text

    def test_reconstruct_story_single_plot(
        self, narrator_engine: NarratorEngine, rng: np.random.Generator
    ):
        """使用单个情节测试reconstruct_story。"""
        plot = make_plot(idx=0, rng=rng, text="单个测试交互")
        trace = narrator_engine.reconstruct_story(
            query="测试查询",
            plots=[plot],
        )

        assert len(trace.elements) == 1
        assert trace.elements[0].plot_id == plot.id
        assert trace.elements[0].content in plot.text or "..." in trace.elements[0].content

    def test_reconstruct_story_multiple_plots(
        self, narrator_engine: NarratorEngine, sample_plots_for_narrator: List[Plot]
    ):
        """使用多个情节测试reconstruct_story。"""
        trace = narrator_engine.reconstruct_story(
            query="测试查询",
            plots=sample_plots_for_narrator,
        )

        assert len(trace.elements) == len(sample_plots_for_narrator)
        assert trace.perspective in NarrativePerspective
        assert trace.reconstruction_confidence > 0
        assert len(trace.narrative_text) > 0

    def test_reconstruct_story_with_specific_perspective(
        self, narrator_engine: NarratorEngine, sample_plots_for_narrator: List[Plot]
    ):
        """使用指定的视角测试reconstruct_story。"""
        for perspective in NarrativePerspective:
            trace = narrator_engine.reconstruct_story(
                query="测试查询",
                plots=sample_plots_for_narrator,
                perspective=perspective,
            )

            assert trace.perspective == perspective
            assert len(trace.elements) > 0

    def test_select_perspective_returns_valid_perspective(
        self, narrator_engine: NarratorEngine, sample_plots_for_narrator: List[Plot]
    ):
        """测试select_perspective返回有效的视角。"""
        perspective, probs = narrator_engine.select_perspective(
            query="如何处理矛盾？",
            plots=sample_plots_for_narrator,
        )

        assert perspective in NarrativePerspective
        assert isinstance(probs, dict)
        assert len(probs) == len(NarrativePerspective)
        assert all(0 <= p <= 1 for p in probs.values())
        # Probabilities should sum to approximately 1
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_select_perspective_without_plots(self, narrator_engine: NarratorEngine):
        """未提供情节时测试select_perspective。"""
        perspective, probs = narrator_engine.select_perspective(
            query="测试查询",
            plots=None,
        )

        assert perspective in NarrativePerspective
        assert len(probs) > 0

    def test_recover_context(
        self, narrator_engine: NarratorEngine, sample_plots_for_narrator: List[Plot]
    ):
        """测试情节的上下文恢复。"""
        plots_dict = {p.id: p for p in sample_plots_for_narrator}
        target_plot = sample_plots_for_narrator[2]

        context_plots = narrator_engine.recover_context(
            plot=target_plot,
            plots_dict=plots_dict,
            depth=2,
        )

        # Context should not include the target plot itself
        assert all(p.id != target_plot.id for p in context_plots)
        # Context should be a subset of available plots
        assert all(p.id in plots_dict for p in context_plots)

    def test_identify_turning_points(
        self, narrator_engine: NarratorEngine, sample_plots_for_narrator: List[Plot]
    ):
        """测试转折点识别。"""
        turning_points = narrator_engine.identify_turning_points(
            plots=sample_plots_for_narrator,
        )

        # All turning points should be from the input plots
        plot_ids = {p.id for p in sample_plots_for_narrator}
        for tp in turning_points:
            assert tp.id in plot_ids

    def test_feedback_narrative(
        self, narrator_engine: NarratorEngine, sample_plots_for_narrator: List[Plot]
    ):
        """测试反馈更新视角信念。"""
        trace = narrator_engine.reconstruct_story(
            query="测试查询",
            plots=sample_plots_for_narrator,
        )

        # Get beliefs before feedback
        beliefs_before = narrator_engine.perspective_beliefs[trace.perspective]

        # Positive feedback
        narrator_engine.feedback_narrative(trace, success=True)

        # Check beliefs updated
        beliefs_after = narrator_engine.perspective_beliefs[trace.perspective]
        assert beliefs_after[0] == beliefs_before[0] + 1.0  # Alpha increased

    def test_feedback_narrative_negative(
        self, narrator_engine: NarratorEngine, sample_plots_for_narrator: List[Plot]
    ):
        """测试负反馈更新视角信念。"""
        trace = narrator_engine.reconstruct_story(
            query="测试查询",
            plots=sample_plots_for_narrator,
        )

        beliefs_before = narrator_engine.perspective_beliefs[trace.perspective]

        # Negative feedback
        narrator_engine.feedback_narrative(trace, success=False)

        beliefs_after = narrator_engine.perspective_beliefs[trace.perspective]
        assert beliefs_after[1] == beliefs_before[1] + 1.0  # Beta increased

    def test_serialization_round_trip(
        self, narrator_metric: LowRankMetric, narrator_engine: NarratorEngine
    ):
        """测试序列化和反序列化。"""
        # Modify state
        narrator_engine.perspective_beliefs[NarrativePerspective.FOCUSED] = (5.0, 2.0)

        # Serialize
        state = narrator_engine.to_state_dict()

        # Deserialize
        restored = NarratorEngine.from_state_dict(
            state,
            metric=narrator_metric,
        )

        # Verify state
        assert restored._seed == narrator_engine._seed
        assert (
            restored.perspective_beliefs[NarrativePerspective.FOCUSED]
            == narrator_engine.perspective_beliefs[NarrativePerspective.FOCUSED]
        )


# =============================================================================
# PerspectiveSelector Tests
# =============================================================================


class TestPerspectiveSelector:
    """PerspectiveSelector类的测试。"""

    def test_select_perspective_deterministic(
        self, narrator_metric: LowRankMetric, sample_plots_for_narrator: List[Plot]
    ):
        """测试相同种子产生可重现的结果。"""
        selector1 = PerspectiveSelector(
            metric=narrator_metric,
            rng=np.random.default_rng(42),
        )
        selector2 = PerspectiveSelector(
            metric=narrator_metric,
            rng=np.random.default_rng(42),
        )

        p1, probs1 = selector1.select_perspective(
            query="测试",
            plots=sample_plots_for_narrator,
        )
        p2, probs2 = selector2.select_perspective(
            query="测试",
            plots=sample_plots_for_narrator,
        )

        assert p1 == p2
        assert probs1 == probs2

    def test_feedback_increases_alpha(
        self, perspective_selector: PerspectiveSelector
    ):
        """测试正反馈增加alpha。"""
        perspective = NarrativePerspective.CHRONOLOGICAL
        a_before, b_before = perspective_selector.perspective_beliefs[perspective]

        perspective_selector.feedback(perspective, success=True)

        a_after, b_after = perspective_selector.perspective_beliefs[perspective]
        assert a_after == a_before + 1.0
        assert b_after == b_before

    def test_feedback_increases_beta(
        self, perspective_selector: PerspectiveSelector
    ):
        """测试负反馈增加beta。"""
        perspective = NarrativePerspective.CONTRASTIVE
        a_before, b_before = perspective_selector.perspective_beliefs[perspective]

        perspective_selector.feedback(perspective, success=False)

        a_after, b_after = perspective_selector.perspective_beliefs[perspective]
        assert a_after == a_before
        assert b_after == b_before + 1.0


# =============================================================================
# PerspectiveOrganizer Tests
# =============================================================================


class TestPerspectiveOrganizer:
    """PerspectiveOrganizer类的测试。"""

    def test_organize_chronological(
        self, perspective_organizer: PerspectiveOrganizer, rng: np.random.Generator
    ):
        """测试时间序组织按时间戳排序。"""
        # 创建具有特定时间戳的情节
        plots = [
            make_plot(idx=i, rng=rng, ts_offset=offset)
            for i, offset in enumerate([300, 100, 200])  # 不按顺序
        ]

        def compute_sig(p: Plot) -> float:
            return 0.5 + 0.1 * p.tension

        elements = perspective_organizer.organize_chronological(plots, compute_sig)

        assert len(elements) == 3
        # Should be sorted by timestamp
        timestamps = [e["timestamp"] for e in elements]
        assert timestamps == sorted(timestamps)

    def test_organize_retrospective(
        self,
        perspective_organizer: PerspectiveOrganizer,
        rng: np.random.Generator,
        narrator_metric: LowRankMetric,
    ):
        """测试回顾式组织优先考虑最近性和相关性。"""
        plots = [make_plot(idx=i, rng=rng, ts_offset=i * 3600) for i in range(3)]
        query_embedding = plots[2].embedding  # Most similar to last plot

        def compute_sig(p: Plot) -> float:
            return 0.5

        elements = perspective_organizer.organize_retrospective(
            plots, "测试", query_embedding, compute_sig
        )

        assert len(elements) == 3
        # First element should have annotation "回顾"
        assert elements[0]["annotation"] == "回顾"

    def test_organize_contrastive(
        self, perspective_organizer: PerspectiveOrganizer, rng: np.random.Generator
    ):
        """测试对比式组织配对对比的情节。"""
        # Create plots with different tensions
        plots = [
            make_plot(idx=0, rng=rng, tension=0.2),
            make_plot(idx=1, rng=rng, tension=0.8),
            make_plot(idx=2, rng=rng, tension=0.3),
        ]

        def compute_sig(p: Plot) -> float:
            return 0.5 + 0.2 * p.tension

        elements = perspective_organizer.organize_contrastive(plots, compute_sig)

        # Should have annotations for contrast
        annotations = [e["annotation"] for e in elements]
        assert any("对比" in a for a in annotations if a)


# =============================================================================
# ContextRecovery Tests
# =============================================================================


class TestContextRecovery:
    """ContextRecovery类的测试。"""

    def test_recover_context_basic(
        self, context_recovery: ContextRecovery, rng: np.random.Generator
    ):
        """测试基本上下文恢复。"""
        # Create plots with temporal proximity
        plots = [make_plot(idx=i, rng=rng, ts_offset=i * 60) for i in range(5)]
        plots_dict = {p.id: p for p in plots}

        # Recover context for the middle plot
        target = plots[2]
        context = context_recovery.recover_context(
            plot=target,
            plots_dict=plots_dict,
            depth=2,
        )

        # Context should not include target
        assert all(p.id != target.id for p in context)

    def test_recover_context_with_story_connection(
        self, context_recovery: ContextRecovery, rng: np.random.Generator
    ):
        """测试具有故事连接的上下文恢复。"""
        story_id = det_id("story", "test_story")
        plots = [
            make_plot(idx=i, rng=rng, ts_offset=i * 60, story_id=story_id)
            for i in range(3)
        ]
        plots_dict = {p.id: p for p in plots}

        target = plots[1]
        context = context_recovery.recover_context(
            plot=target,
            plots_dict=plots_dict,
            depth=2,
        )

        # Should recover some context (probabilistic)
        # Just verify it doesn't crash and returns valid plots
        assert all(p.id in plots_dict for p in context)

    def test_identify_turning_points_few_plots(
        self, context_recovery: ContextRecovery, rng: np.random.Generator
    ):
        """测试具有少量情节的转折点识别。"""
        # With only one plot, should return empty
        plots = [make_plot(idx=0, rng=rng)]
        turning_points = context_recovery.identify_turning_points(plots)
        assert turning_points == []

    def test_identify_turning_points_with_tension_variance(
        self, context_recovery: ContextRecovery, rng: np.random.Generator
    ):
        """测试具有不同张力的转折点识别。"""
        # Create plots with a tension peak
        tensions = [0.2, 0.4, 0.9, 0.5, 0.3]  # Peak at index 2
        plots = [
            make_plot(idx=i, rng=rng, ts_offset=i * 60, tension=t)
            for i, t in enumerate(tensions)
        ]

        turning_points = context_recovery.identify_turning_points(plots)

        # Should identify some turning points (probabilistic)
        plot_ids = {p.id for p in plots}
        for tp in turning_points:
            assert tp.id in plot_ids


# =============================================================================
# TurningPointDetector Tests
# =============================================================================


class TestTurningPointDetector:
    """TurningPointDetector类的测试。"""

    def test_detect_from_elements_empty(
        self, turning_point_detector: TurningPointDetector
    ):
        """测试使用空元素的检测。"""
        turning_points = turning_point_detector.detect_from_elements([])
        assert turning_points == []

    def test_detect_from_elements_single(
        self, turning_point_detector: TurningPointDetector
    ):
        """测试使用单个元素的检测。"""
        elements = [{"plot_id": "p1", "tension_level": 0.5}]
        turning_points = turning_point_detector.detect_from_elements(elements)
        assert turning_points == []  # Need at least 2 for comparison

    def test_detect_from_elements_with_variance(
        self, turning_point_detector: TurningPointDetector
    ):
        """测试具有不同张力水平的检测。"""
        elements = [
            {"plot_id": f"p{i}", "tension_level": t}
            for i, t in enumerate([0.1, 0.3, 0.9, 0.4, 0.2])
        ]

        turning_points = turning_point_detector.detect_from_elements(elements)

        # All turning points should be from input elements
        element_ids = {e["plot_id"] for e in elements}
        for tp in turning_points:
            assert tp["plot_id"] in element_ids


# =============================================================================
# NarrativeElement and NarrativeTrace Tests
# =============================================================================


class TestDataClasses:
    """NarrativeElement和NarrativeTrace数据类的测试。"""

    def test_narrative_element_serialization(self):
        """测试NarrativeElement序列化往返。"""
        element = NarrativeElement(
            plot_id="test_plot",
            role=NarrativeRole.CLIMAX,
            content="测试内容",
            timestamp=now_ts(),
            significance=0.8,
            causes=["cause1", "cause2"],
            effects=["effect1"],
            annotation="转折点",
            tension_level=0.9,
        )

        # Serialize
        state = element.to_state_dict()

        # Deserialize
        restored = NarrativeElement.from_state_dict(state)

        assert restored.plot_id == element.plot_id
        assert restored.role == element.role
        assert restored.content == element.content
        assert restored.significance == element.significance
        assert restored.causes == element.causes
        assert restored.effects == element.effects
        assert restored.annotation == element.annotation
        assert restored.tension_level == element.tension_level

    def test_narrative_trace_serialization(self):
        """测试NarrativeTrace序列化往返。"""
        element = NarrativeElement(
            plot_id="p1",
            role=NarrativeRole.EXPOSITION,
            content="开场",
            timestamp=now_ts(),
        )

        trace = NarrativeTrace(
            query="测试查询",
            perspective=NarrativePerspective.RETROSPECTIVE,
            elements=[element],
            reconstruction_confidence=0.75,
            perspective_probs={"chronological": 0.3, "retrospective": 0.7},
            turning_point_ids=["p1"],
            causal_chain_depth=2,
            narrative_text="测试叙事文本",
        )

        # Serialize
        state = trace.to_state_dict()

        # Deserialize
        restored = NarrativeTrace.from_state_dict(state)

        assert restored.query == trace.query
        assert restored.perspective == trace.perspective
        assert len(restored.elements) == len(trace.elements)
        assert restored.reconstruction_confidence == trace.reconstruction_confidence
        assert restored.perspective_probs == trace.perspective_probs
        assert restored.turning_point_ids == trace.turning_point_ids
        assert restored.narrative_text == trace.narrative_text


# =============================================================================
# Integration Tests
# =============================================================================


class TestNarratorIntegration:
    """叙述者模块的集成测试。"""

    def test_full_narrative_flow(
        self, narrator_engine: NarratorEngine, sample_plots_for_narrator: List[Plot]
    ):
        """测试完整的叙事重构流程。"""
        # Step 1: Select perspective
        perspective, probs = narrator_engine.select_perspective(
            query="发生了什么变化？",
            plots=sample_plots_for_narrator,
        )

        # Step 2: Reconstruct story
        trace = narrator_engine.reconstruct_story(
            query="发生了什么变化？",
            plots=sample_plots_for_narrator,
            perspective=perspective,
        )

        # Step 3: Identify turning points
        turning_points = narrator_engine.identify_turning_points(
            plots=sample_plots_for_narrator,
        )

        # Step 4: Provide feedback
        narrator_engine.feedback_narrative(trace, success=True)

        # Verify the flow completed successfully
        assert trace.perspective == perspective
        assert len(trace.elements) == len(sample_plots_for_narrator)
        assert trace.narrative_text != ""

    def test_narrator_determinism(
        self, narrator_metric: LowRankMetric, sample_plots_for_narrator: List[Plot]
    ):
        """测试叙述者使用相同种子产生确定性结果。"""
        engine1 = NarratorEngine(metric=narrator_metric, seed=42)
        engine2 = NarratorEngine(metric=narrator_metric, seed=42)

        trace1 = engine1.reconstruct_story(
            query="测试",
            plots=sample_plots_for_narrator,
        )
        trace2 = engine2.reconstruct_story(
            query="测试",
            plots=sample_plots_for_narrator,
        )

        assert trace1.perspective == trace2.perspective
        assert len(trace1.elements) == len(trace2.elements)
        for e1, e2 in zip(trace1.elements, trace2.elements):
            assert e1.plot_id == e2.plot_id
