from __future__ import annotations

from aurora.expression.context import ExpressionContext
from aurora.expression.response import plan_response
from aurora.relation.decision import RelationDecisionContext
from aurora.runtime.contracts import TraceChannel


def _ctx(
    *,
    boundary_events: int = 0,
    repair_events: int = 0,
    resonance_events: int = 0,
    thread_count: int = 0,
    knot_count: int = 0,
) -> RelationDecisionContext:
    return RelationDecisionContext(
        boundary_events=boundary_events,
        repair_events=repair_events,
        resonance_events=resonance_events,
        thread_count=thread_count,
        knot_count=knot_count,
    )


def _orientation_snapshot(
    risk_count: int = 0, stability_count: int = 0
) -> dict[str, object]:
    return {
        "self": {"recognition": {"count": 0, "sources": ()}, "fragility": {"count": 0, "sources": ()}, "openness": {"count": 0, "sources": ()}, "agency": {"count": 0, "sources": ()}},
        "world": {
            "welcome": {"count": 0, "sources": ()},
            "risk": {"count": risk_count, "sources": tuple(f"src_{i}" for i in range(risk_count))},
            "mystery": {"count": 0, "sources": ()},
            "stability": {"count": stability_count, "sources": tuple(f"src_{i}" for i in range(stability_count))},
        },
        "relation": {"closeness": {"count": 0, "sources": ()}, "distance": {"count": 0, "sources": ()}, "boundary": {"count": 0, "sources": ()}, "repair": {"count": 0, "sources": ()}},
        "anchor_threads": (),
        "active_knots": (),
    }


def test_high_risk_evidence_pushes_toward_withhold_on_hurt() -> None:
    act = plan_response(
        ExpressionContext(
            input_text="something happened",
            relation_context=_ctx(resonance_events=1),
            dominant_channels=(TraceChannel.HURT, TraceChannel.COHERENCE),
            has_knots=False,
            orientation_snapshot=_orientation_snapshot(risk_count=4),
        )
    )
    assert act.move == "withhold"


def test_no_risk_evidence_allows_witness_on_hurt() -> None:
    act = plan_response(
        ExpressionContext(
            input_text="something happened",
            relation_context=_ctx(resonance_events=1),
            dominant_channels=(TraceChannel.HURT, TraceChannel.COHERENCE),
            has_knots=False,
            orientation_snapshot=_orientation_snapshot(risk_count=0),
        )
    )
    assert act.move == "witness"


def test_orientation_none_does_not_crash() -> None:
    act = plan_response(
        ExpressionContext(
            input_text="hello",
            relation_context=_ctx(),
            dominant_channels=(TraceChannel.COHERENCE,),
            has_knots=False,
            orientation_snapshot=None,
        )
    )
    assert act.move in {"approach", "withhold", "boundary", "repair", "silence", "witness"}
