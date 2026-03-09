from __future__ import annotations

from pathlib import Path

import pytest

from aurora.integrations.llm.provider import LLMProvider
from aurora.integrations.llm.schemas import MemoryBriefCompilation, PlotExtraction
from aurora.runtime.runtime import AuroraRuntime
from aurora.runtime.settings import AuroraSettings


class ContextAwareLLM(LLMProvider):
    def __init__(self):
        self.prompts: list[dict[str, str | None]] = []
        self.memory_compile_calls = 0

    def complete(self, prompt: str, **kwargs) -> str:
        self.prompts.append(
            {
                "prompt": prompt,
                "system": kwargs.get("system"),
            }
        )
        return "这是基于结构化记忆的回复"

    def complete_json(self, *, system: str, user: str, schema, **kwargs):
        if schema.__name__ == "MemoryBriefCompilation":
            self.memory_compile_calls += 1
            return schema.model_validate(
                MemoryBriefCompilation(
                    known_facts=["当前确认：用户偏好长期记忆系统"],
                    preferences=["用户明显偏好长期记忆系统"],
                    relationship_state=["与该用户的关系正在发展，我承担长期记忆助手角色"],
                    active_narratives=["当前叙事围绕长期记忆需求展开"],
                    temporal_context=[],
                    cautions=[],
                ).model_dump()
            )
        return schema.model_validate(
            PlotExtraction(
                actors=["user", "agent"],
                action="表达偏好",
                outcome="系统记录了用户的长期记忆偏好",
                context="memory_preference",
                claims=[
                    {
                        "subject": "user",
                        "predicate": "prefers",
                        "object": "长期记忆系统",
                    }
                ],
            ).model_dump()
        )


class CompilerFailingLLM(ContextAwareLLM):
    def complete_json(self, *, system: str, user: str, schema, **kwargs):
        if schema.__name__ == "MemoryBriefCompilation":
            raise RuntimeError("compiler unavailable")
        return super().complete_json(system=system, user=user, schema=schema, **kwargs)


class DirtyCompilerLLM(ContextAwareLLM):
    def complete_json(self, *, system: str, user: str, schema, **kwargs):
        if schema.__name__ == "MemoryBriefCompilation":
            return schema.model_validate(
                MemoryBriefCompilation(
                    known_facts=["USER: 我喜欢长期记忆系统", "当前确认：用户偏好长期记忆系统"],
                    preferences=["AGENT: 我会记住", "用户明显偏好长期记忆系统"],
                    relationship_state=[],
                    active_narratives=[],
                    temporal_context=[],
                    cautions=[],
                ).model_dump()
            )
        return super().complete_json(system=system, user=user, schema=schema, **kwargs)


@pytest.fixture
def runtime_settings(temp_data_dir: Path) -> AuroraSettings:
    return AuroraSettings(
        data_dir=str(temp_data_dir),
        llm_provider="mock",
        embedding_provider="hash",
        dim=64,
        pii_redaction_enabled=False,
        snapshot_every_events=0,
        memory_seed=7,
    )


def test_build_response_context_collects_structured_sections(runtime_settings: AuroraSettings):
    llm = ContextAwareLLM()
    runtime = AuroraRuntime(settings=runtime_settings, llm=llm)
    runtime.ingest_interaction(
        event_id="evt_ctx_001",
        session_id="chat_ctx",
        user_message="我喜欢长期记忆系统。",
        agent_message="我会记住你偏爱长期记忆系统。",
    )

    memory_context, trace_summary = runtime.build_response_context(
        user_message="你记得我喜欢什么吗？",
        k=6,
    )

    assert any("当前确认" in item for item in memory_context.known_facts)
    assert any("明显偏好长期记忆系统" in item for item in memory_context.preferences)
    assert memory_context.evidence_refs
    assert trace_summary.hit_count >= 1
    assert all(not hasattr(ref, "snippet") for ref in memory_context.evidence_refs)
    assert llm.memory_compile_calls == 1


def test_respond_uses_structured_prompt_without_raw_dialogue(runtime_settings: AuroraSettings):
    llm = ContextAwareLLM()
    runtime = AuroraRuntime(settings=runtime_settings, llm=llm)
    runtime.ingest_interaction(
        event_id="evt_ctx_001",
        session_id="chat_ctx",
        user_message="我喜欢长期记忆系统。",
        agent_message="我会记住你偏爱长期记忆系统。",
    )

    result = runtime.respond(
        session_id="chat_ctx",
        user_message="你记得我喜欢什么吗？",
        k=6,
    )

    prompt = llm.prompts[-1]["prompt"] or ""
    assert result.reply == "这是基于结构化记忆的回复"
    assert result.user_prompt == prompt
    assert "[Known Facts]" in prompt
    assert "Current User Message" in prompt
    assert "USER:" not in prompt
    assert "AGENT:" not in prompt
    assert "我喜欢长期记忆系统。" not in prompt
    assert result.memory_context.evidence_refs
    assert llm.memory_compile_calls == 1


def test_build_response_context_falls_back_to_deterministic_sections(runtime_settings: AuroraSettings):
    runtime = AuroraRuntime(settings=runtime_settings, llm=CompilerFailingLLM())
    runtime.ingest_interaction(
        event_id="evt_ctx_001",
        session_id="chat_ctx",
        user_message="我喜欢长期记忆系统。",
        agent_message="我会记住你偏爱长期记忆系统。",
    )

    memory_context, _ = runtime.build_response_context(
        user_message="你记得我喜欢什么吗？",
        k=6,
    )

    assert any("长期记忆系统" in item for item in memory_context.preferences + memory_context.known_facts)


def test_build_response_context_sanitizes_compiler_output(runtime_settings: AuroraSettings):
    runtime = AuroraRuntime(settings=runtime_settings, llm=DirtyCompilerLLM())
    runtime.ingest_interaction(
        event_id="evt_ctx_001",
        session_id="chat_ctx",
        user_message="我喜欢长期记忆系统。",
        agent_message="我会记住你偏爱长期记忆系统。",
    )

    memory_context, _ = runtime.build_response_context(
        user_message="你记得我喜欢什么吗？",
        k=6,
    )

    joined = "\n".join(memory_context.known_facts + memory_context.preferences)
    assert "USER:" not in joined
    assert "AGENT:" not in joined
    assert "当前确认：用户偏好长期记忆系统" in joined
