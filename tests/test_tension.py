from __future__ import annotations

import json

from tests.conftest import KernelFactory, QueueLLM


def test_open_loop_is_retained_until_resolved(kernel_factory: KernelFactory) -> None:
    llm = QueueLLM(
        "ok",
        json.dumps(
            {
                "ops": [
                    {
                        "type": "loop",
                        "payload": {
                            "state": "active",
                            "loop_type": "commitment",
                            "summary": "提醒用户补上部署文档",
                            "urgency": 0.9,
                        },
                    }
                ]
            },
            ensure_ascii=False,
        ),
        "ok",
        json.dumps({"ops": []}, ensure_ascii=False),
        "ok",
        json.dumps(
            {
                "ops": [
                    {
                        "type": "loop",
                        "payload": {
                            "state": "resolved",
                            "loop_type": "commitment",
                            "summary": "提醒用户补上部署文档",
                        },
                    }
                ]
            },
            ensure_ascii=False,
        ),
    )
    kernel = kernel_factory(llm=llm)

    kernel.turn("relation-d", "之后提醒我补上部署文档。", now_ts=10.0)
    first = kernel.snapshot("relation-d")

    loop = next(
        item
        for item in first.open_loops
        if item.loop_type == "commitment" and item.summary == "提醒用户补上部署文档"
    )
    follow_up = kernel.turn("relation-d", "继续刚才那个承诺。", now_ts=12.0)
    assert follow_up.response_text.strip()
    mid = kernel.snapshot("relation-d")
    assert any(item.loop_id == loop.loop_id and item.status == "active" for item in mid.open_loops)

    kernel.turn("relation-d", "这个承诺已经完成了。", now_ts=20.0)
    second = kernel.snapshot("relation-d")

    resolved = next(item for item in second.open_loops if item.loop_id == loop.loop_id)
    assert resolved.status == "resolved"
    assert resolved.updated_at == 20.0


def test_forget_atom_hides_related_memory_from_main_continuity(kernel_factory: KernelFactory) -> None:
    def _answer_if_recalled(messages: list[dict[str, str]]) -> str:
        system_text = "\n".join(message["content"] for message in messages if message["role"] == "system")
        assert "用户喜欢爵士乐" in system_text
        return "爵士乐"

    llm = QueueLLM(
        "ok",
        json.dumps(
            {
                "ops": [
                    {
                        "type": "fact",
                        "payload": {
                            "content": "用户喜欢爵士乐",
                            "fact_kind": "preference",
                            "confidence": 0.9,
                        },
                    }
                ]
            },
            ensure_ascii=False,
        ),
        _answer_if_recalled,
        json.dumps({"ops": []}, ensure_ascii=False),
    )
    kernel = kernel_factory(llm=llm)
    relation_id = "relation-e"

    kernel.turn(relation_id, "我喜欢爵士乐。", now_ts=1.0)
    target = next(
        fact
        for fact in kernel.snapshot(relation_id).facts
        if fact.status == "active" and "用户喜欢爵士乐" in fact.content
    )

    before = kernel.recall(relation_id, "我喜欢什么音乐？")
    assert any("爵士乐" in hit.content for hit in before.hits)

    before_turn = kernel.turn(relation_id, "我喜欢什么音乐？", now_ts=1.5)
    assert "爵士乐" in before_turn.response_text

    next_llm = QueueLLM(
        "ok",
        json.dumps(
            {
                "ops": [
                    {
                        "type": "forget",
                        "payload": {
                            "matcher": "请忘掉：我喜欢爵士乐。",
                            "target_atom_ids": [target.fact_id],
                            "summary": "forget jazz",
                        },
                    }
                ]
            },
            ensure_ascii=False,
        ),
        "ok",
        json.dumps({"ops": []}, ensure_ascii=False),
    )
    kernel_factory.track(next_llm)  # type: ignore[attr-defined]
    kernel.llm = next_llm
    kernel.turn(relation_id, "请忘掉：我喜欢爵士乐。", now_ts=2.0)

    after_snapshot = kernel.snapshot(relation_id)
    active_contents = [fact.content for fact in after_snapshot.facts if fact.status == "active"]
    assert all("爵士乐" not in content for content in active_contents)

    after = kernel.recall(relation_id, "我喜欢什么音乐？")
    assert all("爵士乐" not in hit.content for hit in after.hits)

    after_turn = kernel.turn(relation_id, "我喜欢什么音乐？", now_ts=3.0)
    assert "爵士乐" not in after_turn.response_text

    kernel.close()
    resumed = kernel_factory(llm=QueueLLM("ok", json.dumps({"ops": []}, ensure_ascii=False)))

    resumed_snapshot = resumed.snapshot(relation_id)
    resumed_active_contents = [fact.content for fact in resumed_snapshot.facts if fact.status == "active"]
    assert all("爵士乐" not in content for content in resumed_active_contents)

    resumed_recall = resumed.recall(relation_id, "我喜欢什么音乐？")
    assert all("爵士乐" not in hit.content for hit in resumed_recall.hits)

    resumed_turn = resumed.turn(relation_id, "我喜欢什么音乐？", now_ts=4.0)
    assert "爵士乐" not in resumed_turn.response_text
