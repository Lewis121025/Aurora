"""Tests for narrator module.

This module tests the narrative reconstruction engine and its components:
- NarratorEngine: Main narrative reconstruction engine
- PerspectiveSelector: Bayesian perspective selection
- PerspectiveOrganizer: Perspective-specific plot organization
- ContextRecovery: Causal context recovery and turning point detection
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
    """Create a LowRankMetric for narrator tests."""
    return LowRankMetric(dim=64, rank=16, seed=42)


@pytest.fixture
def rng() -> np.random.Generator:
    """Create a random generator with fixed seed."""
    return np.random.default_rng(42)


@pytest.fixture
def narrator_engine(narrator_metric: LowRankMetric) -> NarratorEngine:
    """Create a NarratorEngine instance."""
    return NarratorEngine(metric=narrator_metric, seed=42)


@pytest.fixture
def perspective_selector(
    narrator_metric: LowRankMetric, rng: np.random.Generator
) -> PerspectiveSelector:
    """Create a PerspectiveSelector instance."""
    return PerspectiveSelector(metric=narrator_metric, rng=rng)


@pytest.fixture
def perspective_organizer(
    narrator_metric: LowRankMetric, rng: np.random.Generator
) -> PerspectiveOrganizer:
    """Create a PerspectiveOrganizer instance."""
    return PerspectiveOrganizer(metric=narrator_metric, rng=rng)


@pytest.fixture
def context_recovery(
    narrator_metric: LowRankMetric, rng: np.random.Generator
) -> ContextRecovery:
    """Create a ContextRecovery instance."""
    return ContextRecovery(metric=narrator_metric, rng=rng)


@pytest.fixture
def turning_point_detector(rng: np.random.Generator) -> TurningPointDetector:
    """Create a TurningPointDetector instance."""
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
    """Helper to create a test Plot.
    
    Note: ts_offset should be negative or zero to ensure plots are in the past.
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

    # Use negative offset to ensure plots are in the past
    # This prevents math domain errors in log1p(now - plot.ts)
    base_ts = now_ts() - 86400  # Start from 1 day ago
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
    """Create a list of test plots with varying tensions and timestamps."""
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
            ts_offset=i * 3600,  # 1 hour apart (relative to base_ts in make_plot)
            tension=tension,
            text=text,
            with_identity_impact=(i == 2),  # Third plot has identity impact
        )
        plots.append(plot)

    return plots


# =============================================================================
# NarratorEngine Tests
# =============================================================================


class TestNarratorEngine:
    """Tests for NarratorEngine class."""

    def test_reconstruct_story_empty_input(self, narrator_engine: NarratorEngine):
        """Test reconstruct_story with empty plot list."""
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
        """Test reconstruct_story with a single plot."""
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
        """Test reconstruct_story with multiple plots."""
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
        """Test reconstruct_story with a specified perspective."""
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
        """Test that select_perspective returns a valid perspective."""
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
        """Test select_perspective when no plots are provided."""
        perspective, probs = narrator_engine.select_perspective(
            query="测试查询",
            plots=None,
        )

        assert perspective in NarrativePerspective
        assert len(probs) > 0

    def test_recover_context(
        self, narrator_engine: NarratorEngine, sample_plots_for_narrator: List[Plot]
    ):
        """Test context recovery for a plot."""
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
        """Test turning point identification."""
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
        """Test feedback updates perspective beliefs."""
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
        """Test negative feedback updates perspective beliefs."""
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
        """Test serialization and deserialization."""
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
    """Tests for PerspectiveSelector class."""

    def test_select_perspective_deterministic(
        self, narrator_metric: LowRankMetric, sample_plots_for_narrator: List[Plot]
    ):
        """Test that same seed produces reproducible results."""
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
        """Test that positive feedback increases alpha."""
        perspective = NarrativePerspective.CHRONOLOGICAL
        a_before, b_before = perspective_selector.perspective_beliefs[perspective]

        perspective_selector.feedback(perspective, success=True)

        a_after, b_after = perspective_selector.perspective_beliefs[perspective]
        assert a_after == a_before + 1.0
        assert b_after == b_before

    def test_feedback_increases_beta(
        self, perspective_selector: PerspectiveSelector
    ):
        """Test that negative feedback increases beta."""
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
    """Tests for PerspectiveOrganizer class."""

    def test_organize_chronological(
        self, perspective_organizer: PerspectiveOrganizer, rng: np.random.Generator
    ):
        """Test chronological organization orders by timestamp."""
        # Create plots with specific timestamps
        plots = [
            make_plot(idx=i, rng=rng, ts_offset=offset)
            for i, offset in enumerate([300, 100, 200])  # Not in order
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
        """Test retrospective organization prioritizes recency and relevance."""
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
        """Test contrastive organization pairs contrasting plots."""
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
    """Tests for ContextRecovery class."""

    def test_recover_context_basic(
        self, context_recovery: ContextRecovery, rng: np.random.Generator
    ):
        """Test basic context recovery."""
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
        """Test context recovery with story connections."""
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
        """Test turning point identification with few plots."""
        # With only one plot, should return empty
        plots = [make_plot(idx=0, rng=rng)]
        turning_points = context_recovery.identify_turning_points(plots)
        assert turning_points == []

    def test_identify_turning_points_with_tension_variance(
        self, context_recovery: ContextRecovery, rng: np.random.Generator
    ):
        """Test turning point identification with varying tension."""
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
    """Tests for TurningPointDetector class."""

    def test_detect_from_elements_empty(
        self, turning_point_detector: TurningPointDetector
    ):
        """Test detection with empty elements."""
        turning_points = turning_point_detector.detect_from_elements([])
        assert turning_points == []

    def test_detect_from_elements_single(
        self, turning_point_detector: TurningPointDetector
    ):
        """Test detection with single element."""
        elements = [{"plot_id": "p1", "tension_level": 0.5}]
        turning_points = turning_point_detector.detect_from_elements(elements)
        assert turning_points == []  # Need at least 2 for comparison

    def test_detect_from_elements_with_variance(
        self, turning_point_detector: TurningPointDetector
    ):
        """Test detection with varying tension levels."""
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
    """Tests for NarrativeElement and NarrativeTrace data classes."""

    def test_narrative_element_serialization(self):
        """Test NarrativeElement serialization round-trip."""
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
        """Test NarrativeTrace serialization round-trip."""
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
    """Integration tests for the narrator module."""

    def test_full_narrative_flow(
        self, narrator_engine: NarratorEngine, sample_plots_for_narrator: List[Plot]
    ):
        """Test the full narrative reconstruction flow."""
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
        """Test that narrator produces deterministic results with same seed."""
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
