from __future__ import annotations

from pathlib import Path

import pytest

from aurora.expression.cognition import LLMCallError, _build_messages, run_cognition
from aurora.expression.context import ExpressionContext
from aurora.relation.decision import RelationDecisionContext
from aurora.runtime.contracts import TraceChannel
from aurora.runtime.engine import AuroraEngine

from tests.conftest import ContextAwareLLM, StubLLM


def _ctx() -> RelationDecisionContext:
    return RelationDecisionContext(
        boundary_events=0, repair_events=0, resonance_events=1,
        thread_count=0, knot_count=0,
    )


def test_cognition_returns_valid_result_from_stub() -> None:
    context = ExpressionContext(
        input_text="hello",
        relation_context=_ctx(),
        dominant_channels=(TraceChannel.WARMTH,),
        has_knots=False,
    )
    result = run_cognition(context, StubLLM())

    assert result is not None
    assert result.move == "witness"
    assert result.response_text
    assert result.touch_channels


def test_context_aware_llm_maps_boundary_input() -> None:
    context = ExpressionContext(
        input_text="不要继续，停",
        relation_context=_ctx(),
        dominant_channels=(),
        has_knots=False,
    )
    result = run_cognition(context, ContextAwareLLM())

    assert result is not None
    assert result.move == "boundary"


class _FailingLLM:
    def complete(self, messages: list[dict[str, str]]) -> str:
        raise ConnectionError("simulated network failure")


def test_llm_provider_failure_raises_typed_error() -> None:
    context = ExpressionContext(
        input_text="hello",
        relation_context=_ctx(),
        dominant_channels=(),
        has_knots=False,
    )
    with pytest.raises(LLMCallError, match="simulated network failure"):
        run_cognition(context, _FailingLLM())


def test_engine_requires_llm(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="requires an LLM provider"):
        AuroraEngine.create(data_dir=str(tmp_path))


def test_engine_turn_with_injected_llm(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path), llm=StubLLM())
    output = engine.handle_turn(session_id="s1", text="hello")
    assert output.response_text
    assert output.aurora_move in {"approach", "withhold", "boundary", "repair", "silence", "witness"}
