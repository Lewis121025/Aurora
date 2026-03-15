from __future__ import annotations

import json

import pytest

from aurora.expression.cognition import (
    CognitionResult,
    LLMEmptyResponseError,
    LLMResponseParseError,
    _build_messages,
    _parse_response,
    run_cognition,
)
from aurora.expression.context import ExpressionContext
from aurora.relation.decision import RelationDecisionContext
from aurora.runtime.contracts import AssocKind, TraceChannel

from tests.conftest import ContextAwareLLM


def _ctx(
    *,
    text: str = "hello",
    resonance: int = 0,
    boundary: int = 0,
) -> ExpressionContext:
    return ExpressionContext(
        input_text=text,
        relation_context=RelationDecisionContext(
            boundary_events=boundary,
            repair_events=0,
            resonance_events=resonance,
            thread_count=0,
            knot_count=0,
        ),
        dominant_channels=(),
        has_knots=False,
    )


def test_parse_valid_json() -> None:
    raw = json.dumps({
        "move": "approach",
        "touch": [{"channel": "warmth", "intensity": 0.7}],
        "response": "I feel close to this.",
    })
    result = _parse_response(raw)
    assert result is not None
    assert result.move == "approach"
    assert result.touch_channels == ((TraceChannel.WARMTH, 0.7),)
    assert result.response_text == "I feel close to this."
    assert result.association_kind == AssocKind.RELATION


def test_parse_boundary_move_maps_to_boundary_assoc() -> None:
    raw = json.dumps({
        "move": "boundary",
        "touch": [{"channel": "boundary", "intensity": 0.8}],
        "response": "No.",
    })
    result = _parse_response(raw)
    assert result is not None
    assert result.move == "boundary"
    assert result.association_kind == AssocKind.BOUNDARY
    assert result.fragment_unresolvedness == 0.24


def test_parse_invalid_json_raises() -> None:
    with pytest.raises(LLMResponseParseError, match="Invalid JSON"):
        _parse_response("not json at all")


def test_parse_empty_response_raises() -> None:
    raw = json.dumps({"move": "approach", "touch": [], "response": ""})
    with pytest.raises(LLMEmptyResponseError, match="empty response"):
        _parse_response(raw)


def test_parse_unknown_move_defaults_to_witness() -> None:
    raw = json.dumps({
        "move": "nonsense",
        "touch": [{"channel": "coherence", "intensity": 0.3}],
        "response": "I am here.",
    })
    result = _parse_response(raw)
    assert result is not None
    assert result.move == "witness"


def test_parse_unknown_channel_is_skipped() -> None:
    raw = json.dumps({
        "move": "approach",
        "touch": [{"channel": "fake_channel", "intensity": 0.5}],
        "response": "hello",
    })
    result = _parse_response(raw)
    assert result is not None
    assert result.touch_channels == ((TraceChannel.COHERENCE, 0.26),)


def test_parse_markdown_wrapped_json() -> None:
    raw = '```json\n{"move":"witness","touch":[{"channel":"warmth","intensity":0.4}],"response":"yes"}\n```'
    result = _parse_response(raw)
    assert result is not None
    assert result.move == "witness"


def test_parse_intensity_clamped() -> None:
    raw = json.dumps({
        "move": "approach",
        "touch": [{"channel": "warmth", "intensity": 5.0}],
        "response": "hi",
    })
    result = _parse_response(raw)
    assert result is not None
    assert result.touch_channels[0][1] == 1.0


def test_build_messages_minimal() -> None:
    messages = _build_messages(_ctx(text="hi"))
    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": "hi"}


def test_context_recalled_surfaces_influence_llm_decision() -> None:
    base = RelationDecisionContext(
        boundary_events=0, repair_events=0, resonance_events=0,
        thread_count=0, knot_count=0,
    )
    ctx_without = ExpressionContext(
        input_text="hello",
        relation_context=base,
        dominant_channels=(),
        has_knots=False,
    )
    ctx_with_recall = ExpressionContext(
        input_text="hello",
        relation_context=base,
        dominant_channels=(),
        has_knots=False,
        recalled_surfaces=("a past conversation",),
    )
    llm = ContextAwareLLM()
    result_without = run_cognition(ctx_without, llm)
    result_with = run_cognition(ctx_with_recall, llm)
    assert result_without.move == "witness"
    assert result_with.move == "approach"


def test_context_orientation_risk_influences_llm_decision() -> None:
    base = RelationDecisionContext(
        boundary_events=0, repair_events=0, resonance_events=0,
        thread_count=0, knot_count=0,
    )
    ctx_no_risk = ExpressionContext(
        input_text="hello",
        relation_context=base,
        dominant_channels=(),
        has_knots=False,
    )
    ctx_with_risk = ExpressionContext(
        input_text="hello",
        relation_context=base,
        dominant_channels=(),
        has_knots=False,
        orientation_snapshot={
            "world": {"risk": {"count": 3, "sources": ["s1", "s2", "s3"]}},
            "self": {},
        },
    )
    llm = ContextAwareLLM()
    result_no_risk = run_cognition(ctx_no_risk, llm)
    result_with_risk = run_cognition(ctx_with_risk, llm)
    assert result_no_risk.move == "witness"
    assert result_with_risk.move == "withhold"


def test_build_messages_includes_context() -> None:
    ctx = ExpressionContext(
        input_text="test",
        relation_context=RelationDecisionContext(
            boundary_events=2,
            repair_events=1,
            resonance_events=3,
            thread_count=1,
            knot_count=0,
        ),
        dominant_channels=(TraceChannel.WARMTH, TraceChannel.HURT),
        has_knots=True,
        recalled_surfaces=("memory one", "memory two"),
        recent_summaries=("exchange summary",),
    )
    messages = _build_messages(ctx)
    context_parts = [m["content"] for m in messages if m["role"] == "system"]
    joined = "\n".join(context_parts)
    assert "memory one" in joined
    assert "exchange summary" in joined
    assert "warmth" in joined
    assert "Unresolved tension knots" in joined
    assert "3 resonance" in joined
