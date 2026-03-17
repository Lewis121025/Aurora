from __future__ import annotations

import json

from tests.conftest import KernelFactory, ScriptedLLM


def test_relation_rules_persist_and_shape_future_turns(kernel_factory: KernelFactory) -> None:
    llm = ScriptedLLM(
        compiler_outputs=(
            json.dumps(
                {
                    "ops": [
                        {
                            "type": "patch_relation",
                            "payload": {
                                "trust_delta": 0.18,
                                "distance_delta": -0.12,
                                "warmth_delta": 0.08,
                            },
                        },
                        {
                            "type": "add_rule",
                            "payload": {
                                "rule": "不要安抚式表达，直接一点。",
                                "reason": "explicit_feedback",
                            },
                        },
                    ]
                },
                ensure_ascii=False,
            ),
        )
    )
    kernel = kernel_factory(llm=llm)

    first_turn = kernel.turn("session-a", "以后别用安抚式表达，直接一点。", now_ts=100.0)
    assert first_turn.recall_used is False

    report = kernel.compile_pending("session-a", now_ts=101.0)
    assert report.failures == ()

    initial = kernel.snapshot("session-a")
    assert initial.field.trust > 0.35
    assert initial.field.distance < 0.65
    assert "不要安抚式表达，直接一点。" in initial.field.interaction_rules

    kernel.close()

    resumed = kernel_factory(llm=ScriptedLLM())
    follow_up = resumed.turn("session-a", "继续，我们直接看问题。", now_ts=200.0)
    assert "直接回答" in follow_up.response_text

    restored = resumed.snapshot("session-a")
    assert restored.field.interaction_rules == initial.field.interaction_rules
    assert restored.field.trust == initial.field.trust


def test_compile_failure_keeps_state_and_pending_events(kernel_factory: KernelFactory) -> None:
    llm = ScriptedLLM(
        compiler_outputs=(
            json.dumps(
                {
                    "ops": [
                        {
                            "type": "add_rule",
                            "payload": {
                                "rule": "请直接指出问题。",
                                "reason": "explicit_feedback",
                            },
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            "{bad json",
        )
    )
    kernel = kernel_factory(llm=llm)

    kernel.turn("session-b", "请记住：以后直接指出问题。", now_ts=10.0)
    first_report = kernel.compile_pending("session-b", now_ts=11.0)
    assert first_report.failures == ()

    before = kernel.snapshot("session-b")
    assert before.pending_compile_count == 0
    assert before.field.interaction_rules == ["请直接指出问题。"]

    kernel.turn("session-b", "再提醒你一次。", now_ts=12.0)
    failed = kernel.compile_pending("session-b", now_ts=13.0)
    assert failed.failures

    after = kernel.snapshot("session-b")
    assert after.field.interaction_rules == before.field.interaction_rules
    assert after.pending_compile_count == 2
