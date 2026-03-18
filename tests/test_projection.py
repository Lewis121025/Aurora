from __future__ import annotations

import json
import re

from tests.conftest import KernelFactory, QueueLLM


def test_natural_recall_uses_active_fact_without_session_boundaries(kernel_factory: KernelFactory) -> None:
    def _cognition_expect_recall(messages: list[dict[str, str]]) -> str:
        system_text = "\n".join(message["content"] for message in messages if message["role"] == "system")
        assert "用户现在住在杭州" in system_text
        return "杭州"

    def _cognition_expect_no_recall(messages: list[dict[str, str]]) -> str:
        system_text = "\n".join(message["content"] for message in messages if message["role"] == "system")
        assert "用户现在住在杭州" not in system_text
        return "ok"

    llm = QueueLLM(
        "ok",
        json.dumps(
            {
                "ops": [
                    {
                        "type": "fact",
                        "payload": {
                            "content": "用户现在住在杭州",
                            "fact_kind": "current_state",
                            "confidence": 0.95,
                        },
                    }
                ]
            },
            ensure_ascii=False,
        ),
        _cognition_expect_recall,
        json.dumps({"ops": []}, ensure_ascii=False),
        _cognition_expect_no_recall,
        json.dumps({"ops": []}, ensure_ascii=False),
    )
    kernel = kernel_factory(llm=llm)

    kernel.turn("relation-b", "我现在住在杭州。", now_ts=1.0)

    recalled = kernel.turn("relation-b", "我现在住在哪里？", now_ts=2.0)
    assert recalled.recall_used is True
    assert recalled.recalled_ids
    assert "杭州" in recalled.response_text

    plain = kernel.turn("relation-b", "你好。", now_ts=3.0)
    assert plain.recall_used is False
    assert plain.recalled_ids == ()
    assert plain.response_text.strip()


def test_revision_supersedes_old_fact_and_future_recall_prefers_new_one(kernel_factory: KernelFactory) -> None:
    initial_llm = QueueLLM(
        "ok",
        json.dumps(
            {
                "ops": [
                    {
                        "type": "fact",
                        "payload": {
                            "content": "用户住在上海",
                            "fact_kind": "current_state",
                            "confidence": 0.90,
                        },
                    }
                ]
            },
            ensure_ascii=False,
        ),
    )
    kernel = kernel_factory(llm=initial_llm)

    kernel.turn("relation-c", "我住在上海。", now_ts=100.0)
    original = next(
        fact
        for fact in kernel.snapshot("relation-c").facts
        if fact.status == "active" and "用户住在上海" in fact.content
    )

    next_llm = QueueLLM(
        "ok",
        json.dumps(
            {
                "ops": [
                    {
                        "type": "revision",
                        "payload": {
                            "target_atom_id": original.fact_id,
                            "content": "用户现在住在杭州",
                            "fact_kind": "current_state",
                            "summary": "居住地发生更正",
                            "confidence": 0.96,
                        },
                    }
                ]
            },
            ensure_ascii=False,
        ),
        lambda messages: ("杭州" if "杭州" in "\n".join(m["content"] for m in messages if m["role"] == "system") else "ok"),
        json.dumps({"ops": []}, ensure_ascii=False),
    )
    kernel_factory.track(next_llm)  # type: ignore[attr-defined]
    kernel.llm = next_llm

    response = kernel.turn("relation-c", "更正一下，我现在住在杭州。", now_ts=200.0)

    snapshot = kernel.snapshot("relation-c")
    active_contents = [fact.content for fact in snapshot.facts if fact.status == "active"]
    assert any("用户现在住在杭州" in content for content in active_contents)
    assert all("用户住在上海" not in content for content in active_contents)

    recalled = kernel.recall("relation-c", "我现在住在哪里？")
    assert any("杭州" in hit.content for hit in recalled.hits)
    assert all("上海" not in hit.content for hit in recalled.hits)

    turn = kernel.turn("relation-c", "我现在住在哪里？", now_ts=201.0)
    assert "杭州" in turn.response_text
    assert "上海" not in turn.response_text

    kernel.close()
    resumed = kernel_factory(
        llm=QueueLLM(
            lambda messages: ("杭州" if "杭州" in "\n".join(m["content"] for m in messages if m["role"] == "system") else "ok"),
            json.dumps({"ops": []}, ensure_ascii=False),
        )
    )

    resumed_snapshot = resumed.snapshot("relation-c")
    resumed_active_contents = [fact.content for fact in resumed_snapshot.facts if fact.status == "active"]
    assert any("用户现在住在杭州" in content for content in resumed_active_contents)
    assert all("用户住在上海" not in content for content in resumed_active_contents)

    resumed_recalled = resumed.recall("relation-c", "我现在住在哪里？")
    assert any("杭州" in hit.content for hit in resumed_recalled.hits)
    assert all("上海" not in hit.content for hit in resumed_recalled.hits)

    resumed_turn = resumed.turn("relation-c", "我现在住在哪里？", now_ts=202.0)
    assert "杭州" in resumed_turn.response_text
    assert "上海" not in resumed_turn.response_text


def test_recall_never_returns_raw_event_hits(kernel_factory: KernelFactory) -> None:
    llm = QueueLLM(
        "ok",
        json.dumps(
            {
                "ops": [
                    {
                        "type": "fact",
                        "payload": {
                            "content": "deadline friday",
                            "fact_kind": "current_state",
                            "confidence": 0.95,
                        },
                    }
                ]
            },
            ensure_ascii=False,
        ),
        "ok",
        json.dumps({"ops": []}, ensure_ascii=False),
    )
    kernel = kernel_factory(llm=llm)
    relation_id = "relation-recall-no-events"

    # Ensure there are evidence events that could be recalled if event recall existed,
    # while also guaranteeing at least one atom hit.
    kernel.turn(relation_id, "deadline is Friday", now_ts=1.0)
    kernel.turn(relation_id, "ok", now_ts=2.0)

    recalled = kernel.recall(relation_id, "deadline friday?")
    assert recalled.hits
    assert any(hit.kind == "atom" for hit in recalled.hits)
    assert all(hit.kind != "event" for hit in recalled.hits)


def test_recall_prefers_atoms_and_still_never_returns_events(kernel_factory: KernelFactory) -> None:
    llm = QueueLLM(
        "ok",
        json.dumps(
            {
                "ops": [
                    {
                        "type": "fact",
                        "payload": {
                            "content": "deadline friday",
                            "fact_kind": "current_state",
                            "confidence": 0.95,
                        },
                    }
                ]
            },
            ensure_ascii=False,
        ),
    )
    kernel = kernel_factory(llm=llm)
    relation_id = "relation-recall-atom-only"

    kernel.turn(relation_id, "deadline friday", now_ts=1.0)
    recalled = kernel.recall(relation_id, "deadline friday?")

    assert recalled.hits
    assert all(hit.kind == "atom" for hit in recalled.hits)


def test_cognition_prompt_does_not_include_recent_turn_transcript(kernel_factory: KernelFactory) -> None:
    user_a = "USER_UNIQUE_aa9c3e0a"
    user_b = "USER_UNIQUE_87f7b4f3"
    assistant_a = "ASSIST_UNIQUE_1b8d4f5d"
    assistant_b = "ASSIST_UNIQUE_4e2f3a91"

    def _no_leak(messages: list[dict[str, str]]) -> str:
        system_text = "\n".join(message["content"] for message in messages if message["role"] == "system")
        assert user_a not in system_text
        assert user_b not in system_text
        assert assistant_a not in system_text
        assert assistant_b not in system_text
        return "ok"

    kernel = kernel_factory(
        llm=QueueLLM(
            assistant_a,
            json.dumps({"ops": []}, ensure_ascii=False),
            assistant_b,
            json.dumps({"ops": []}, ensure_ascii=False),
            _no_leak,
            json.dumps({"ops": []}, ensure_ascii=False),
        )
    )
    relation_id = "relation-cognition-no-transcript"

    kernel.turn(relation_id, user_a, now_ts=1.0)
    kernel.turn(relation_id, user_b, now_ts=2.0)

    snapshot = kernel.snapshot(relation_id)
    assert len(snapshot.recent_events) >= 4

    third = kernel.turn(relation_id, "gamma", now_ts=3.0)
    assert third.response_text == "ok"
