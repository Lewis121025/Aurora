from __future__ import annotations

import json

from tests.conftest import KernelFactory, ScriptedLLM


def test_open_loop_is_retained_until_resolved(kernel_factory: KernelFactory) -> None:
    llm = ScriptedLLM(
        compiler_outputs=(
            json.dumps(
                {
                    "ops": [
                        {
                            "type": "open_loop",
                            "payload": {
                                "loop_type": "commitment",
                                "summary": "提醒用户补上部署文档",
                                "urgency": 0.9,
                            },
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "ops": [
                        {
                            "type": "resolve_loop",
                            "payload": {
                                "loop_type": "commitment",
                                "summary": "提醒用户补上部署文档",
                            },
                        }
                    ]
                },
                ensure_ascii=False,
            ),
        )
    )
    kernel = kernel_factory(llm=llm)

    kernel.turn("session-e", "之后提醒我补上部署文档。", now_ts=10.0)
    kernel.compile_pending("session-e", now_ts=11.0)
    first = kernel.snapshot("session-e")

    loop = next(item for item in first.open_loops if item.loop_type == "commitment")
    follow_up = kernel.turn("session-e", "继续刚才那个承诺。", now_ts=12.0)
    assert "提醒用户补上部署文档" in follow_up.response_text

    kernel.turn("session-e", "这个承诺已经完成了。", now_ts=20.0)
    kernel.compile_pending("session-e", now_ts=21.0)
    second = kernel.snapshot("session-e")

    resolved = next(item for item in second.open_loops if item.loop_id == loop.loop_id)
    assert resolved.status == "resolved"
    assert resolved.updated_at == 21.0
