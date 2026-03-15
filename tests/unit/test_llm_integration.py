from __future__ import annotations

from pathlib import Path

from aurora.expression.context import ExpressionContext
from aurora.expression.prompt import build_messages
from aurora.expression.render import render_response
from aurora.expression.response import ResponseAct, plan_response
from aurora.relation.decision import RelationDecisionContext
from aurora.runtime.contracts import TraceChannel
from aurora.runtime.engine import AuroraEngine


def _ctx() -> RelationDecisionContext:
    return RelationDecisionContext(
        boundary_events=0, repair_events=0, resonance_events=1,
        thread_count=0, knot_count=0,
    )


def test_no_llm_falls_back_to_template() -> None:
    context = ExpressionContext(
        input_text="hello",
        relation_context=_ctx(),
        dominant_channels=(TraceChannel.WARMTH, TraceChannel.RECOGNITION),
        has_knots=False,
    )
    act = plan_response(context)
    rendered = render_response(context, act, llm=None)

    assert rendered.text
    assert rendered.move in {"approach", "witness", "withhold", "boundary", "repair", "silence"}


def test_prompt_build_includes_system_and_user() -> None:
    context = ExpressionContext(
        input_text="谢谢你理解我",
        relation_context=_ctx(),
        dominant_channels=(TraceChannel.WARMTH,),
        has_knots=False,
        recalled_surfaces=("之前的记忆片段",),
        recent_summaries=("之前的交流摘要",),
    )
    act = ResponseAct(
        move="approach",
        tone="gentle",
        response_channels=(TraceChannel.WARMTH, TraceChannel.RECOGNITION),
    )
    messages = build_messages(context, act)

    assert any(m["role"] == "system" for m in messages)
    assert any(m["role"] == "user" for m in messages)
    assert messages[-1]["content"] == "谢谢你理解我"


def test_engine_creates_without_llm_when_env_unset(tmp_path: Path) -> None:
    engine = AuroraEngine.create(data_dir=str(tmp_path))
    assert engine.llm is None
    output = engine.handle_turn(session_id="s1", text="hello")
    assert output.response_text
