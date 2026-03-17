from __future__ import annotations

import json

from tests.conftest import KernelFactory, ScriptedLLM


def test_archive_recall_is_explicit_and_skips_small_talk(kernel_factory: KernelFactory) -> None:
    llm = ScriptedLLM(
        compiler_outputs=(
            json.dumps(
                {
                    "ops": [
                        {
                            "type": "assert_fact",
                            "payload": {
                                "content": "用户现在住在杭州",
                                "confidence": 0.95,
                            },
                        }
                    ]
                },
                ensure_ascii=False,
            ),
        )
    )
    kernel = kernel_factory(llm=llm)

    kernel.turn("session-c", "我现在住在杭州。", now_ts=1.0)
    kernel.compile_pending("session-c", now_ts=2.0)

    recalled = kernel.turn("session-c", "我现在住在哪里？", now_ts=3.0)
    assert recalled.recall_used is True
    assert recalled.recalled_ids
    assert "杭州" in recalled.response_text

    plain = kernel.turn("session-c", "你好。", now_ts=4.0)
    assert plain.recall_used is False
    assert plain.recalled_ids == ()


def test_fact_revision_creates_version_chain_and_contradiction_loop(kernel_factory: KernelFactory) -> None:
    initial_llm = ScriptedLLM(
        compiler_outputs=(
            json.dumps(
                {
                    "ops": [
                        {
                            "type": "assert_fact",
                            "payload": {
                                "content": "用户住在上海",
                                "confidence": 0.9,
                            },
                        }
                    ]
                },
                ensure_ascii=False,
            ),
        )
    )
    kernel = kernel_factory(llm=initial_llm)

    kernel.turn("session-d", "我住在上海。", now_ts=100.0)
    kernel.compile_pending("session-d", now_ts=101.0)
    original = kernel.snapshot("session-d").facts[0]

    kernel.llm = ScriptedLLM(
        compiler_outputs=(
            json.dumps(
                {
                    "ops": [
                        {
                            "type": "revise_fact",
                            "payload": {
                                "target_fact_id": original.fact_id,
                                "content": "用户现在住在杭州",
                                "summary": "居住地发生更正",
                                "confidence": 0.96,
                            },
                        }
                    ]
                },
                ensure_ascii=False,
            ),
        )
    )

    kernel.turn("session-d", "更正一下，我现在住在杭州。", now_ts=200.0)
    kernel.compile_pending("session-d", now_ts=201.0)
    snapshot = kernel.snapshot("session-d")

    active = [fact for fact in snapshot.facts if fact.status == "active"]
    superseded = [fact for fact in snapshot.facts if fact.status == "superseded"]
    contradictions = [loop for loop in snapshot.open_loops if loop.loop_type == "contradiction"]

    assert len(active) == 1
    assert len(superseded) == 1
    assert active[0].supersedes == superseded[0].fact_id
    assert contradictions
    assert contradictions[0].status == "active"
