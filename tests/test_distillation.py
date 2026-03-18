from __future__ import annotations

import json
import re

from tests.conftest import KernelFactory, QueueLLM


def test_relation_rules_persist_and_shape_future_turns(kernel_factory: KernelFactory) -> None:
    kernel = kernel_factory(llm=QueueLLM("ok", json.dumps({"ops": []}, ensure_ascii=False)))

    first_turn = kernel.turn("relation-a", "以后别用安抚式表达，直接一点。", now_ts=100.0)
    assert first_turn.applied_atom_ids

    initial = kernel.snapshot("relation-a")
    assert initial.field.trust > 0.35
    assert initial.field.distance < 0.65
    assert "以后别用安抚式表达，直接一点。" in initial.field.interaction_rules

    kernel.close()

    expected_rule = initial.field.interaction_rules[0]

    def _rule_tokens(rule: str) -> list[str]:
        # Avoid coupling to the exact prompt formatting: require only that some
        # salient pieces of the rule make it into cognition context.
        tokens = [
            token
            for token in re.split(r"[^0-9A-Za-z\u4e00-\u9fff]+", rule)
            if len(token) >= 2
        ]
        if tokens:
            return tokens
        if len(rule) >= 2:
            return [rule[:2]]
        return [rule]

    def _cognition(messages: list[dict[str, str]]) -> str:
        system_text = "\n".join(message["content"] for message in messages if message["role"] == "system")
        tokens = _rule_tokens(expected_rule)
        assert any(token and token in system_text for token in tokens)
        return "ok"

    resumed = kernel_factory(
        llm=QueueLLM(
            _cognition,
            json.dumps({"ops": []}, ensure_ascii=False),
        )
    )
    follow_up = resumed.turn("relation-a", "继续，我们直接看问题。", now_ts=200.0)
    assert follow_up.response_text == "ok"

    restored = resumed.snapshot("relation-a")
    assert restored.field.interaction_rules == initial.field.interaction_rules
    assert restored.field.trust == initial.field.trust


def test_relation_field_is_not_driven_by_recent_transcript(kernel_factory: KernelFactory) -> None:
    steps: list[object] = []
    for _ in range(7):
        steps.append("ok")
        steps.append(json.dumps({"ops": []}, ensure_ascii=False))
    kernel = kernel_factory(llm=QueueLLM(*steps))  # type: ignore[arg-type]
    relation_id = "relation-field-stability"

    kernel.turn(relation_id, "ping", now_ts=1.0)
    baseline = kernel.snapshot(relation_id).field

    for idx in range(2, 8):
        kernel.turn(relation_id, f"ping-{idx}", now_ts=float(idx))

    later = kernel.snapshot(relation_id).field
    assert later == baseline
