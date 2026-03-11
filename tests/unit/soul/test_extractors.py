from __future__ import annotations

import numpy as np

from aurora.integrations.embeddings.hash import HashEmbedding
from aurora.integrations.llm.Prompt.generative_soul_prompt import (
    build_gen_soul_meaning_user_prompt,
    build_gen_soul_semantic_projection_user_prompt,
)
from aurora.integrations.llm.provider import LLMProvider
from aurora.integrations.llm.schemas import MeaningFramePayloadV4, SemanticProjectionPayload
from aurora.soul.extractors import LLMMeaningProvider, LLMNarrativeProvider
from aurora.soul.models import HOMEOSTATIC_AXES, IdentityState, ImagePart, Message, TextPart, schema_from_profile


class RecordingLLM(LLMProvider):
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def complete(self, messages, **kwargs):  # pragma: no cover - not used in these tests
        raise NotImplementedError

    def complete_json(self, *, messages, schema, **kwargs):
        self.calls.append(
            {
                "messages": list(messages),
                "schema": schema,
                "kwargs": kwargs,
            }
        )
        payload = self._responses.pop(0)
        return schema.model_validate(payload)


class ScriptedLLM(LLMProvider):
    def __init__(self, steps: list[object]) -> None:
        self._steps = list(steps)
        self.calls: list[dict[str, object]] = []

    def complete(self, messages, **kwargs):  # pragma: no cover - not used in these tests
        raise NotImplementedError

    def complete_json(self, *, messages, schema, **kwargs):
        self.calls.append(
            {
                "messages": list(messages),
                "schema": schema,
                "kwargs": kwargs,
            }
        )
        step = self._steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return schema.model_validate(step)


def _multimodal_messages() -> list[Message]:
    return [
        Message(
            role="user",
            parts=(
                TextPart(text="他愤怒地砸桌子，像是在威胁我。"),
                ImagePart(uri="https://example.com/angry-table.jpg", mime_type="image/jpeg"),
            ),
        )
    ]


def _identity_state() -> IdentityState:
    axis_state = {name: 0.2 for name, _, _, _ in HOMEOSTATIC_AXES}
    self_vector = np.ones(64, dtype=np.float32)
    self_vector /= np.linalg.norm(self_vector)
    return IdentityState(
        self_vector=self_vector,
        axis_state=axis_state,
        intuition_axes={name: 0.0 for name in axis_state},
        current_mode_label="origin",
    )


def test_semantic_projection_prompt_mentions_visual_signals() -> None:
    prompt = build_gen_soul_semantic_projection_user_prompt(message_count=2, image_part_count=1)

    assert "图片 part 数：1" in prompt
    assert "姿态、表情、环境" in prompt
    assert "危险/照护线索" in prompt
    assert "保留冲突" in prompt


def test_meaning_prompt_mentions_visual_scoring_rules() -> None:
    prompt = build_gen_soul_meaning_user_prompt(
        text="一个人愤怒地砸桌子，场面紧张。",
        axis_names=("affiliation", "vigilance"),
        recent_tags=("anger",),
        message_count=1,
        image_part_count=1,
    )

    assert "图片 part 数：1" in prompt
    assert "必须综合原始图文消息" in prompt
    assert "表情、姿态、距离、环境压迫" in prompt
    assert "valence" in prompt
    assert "threat" in prompt


def test_llm_meaning_provider_project_uses_multimodal_projection_prompt() -> None:
    llm = RecordingLLM(
        responses=[
            {
                "semantic_text": "一个人愤怒地砸桌子，室内气氛紧张，带有明显威胁感。",
            }
        ]
    )
    provider = LLMMeaningProvider(llm=llm)

    projected = provider.project(_multimodal_messages())

    assert projected == "一个人愤怒地砸桌子，室内气氛紧张，带有明显威胁感。"
    assert len(llm.calls) == 1
    call = llm.calls[0]
    messages = call["messages"]
    assert isinstance(messages, list)
    assert "可检索的 semantic_text" in messages[0].parts[0].text
    assert any(part.type == "image" for message in messages for part in message.parts)
    assert "图片 part 数：1" in messages[-1].parts[0].text


def test_llm_meaning_provider_extract_includes_visual_scoring_instructions() -> None:
    llm = RecordingLLM(
        responses=[
            {
                "semantic_text": "一个人愤怒地砸桌子，室内对峙明显，视觉上具有压迫感。",
            },
            {
                "axis_evidence": {
                    "vigilance": 0.7,
                    "affiliation": -0.2,
                },
                "valence": -0.7,
                "arousal": 0.9,
                "care": 0.0,
                "threat": 0.85,
                "control": 0.35,
                "abandonment": 0.0,
                "agency_signal": 0.2,
                "shame": 0.0,
                "novelty": 0.45,
                "self_relevance": 0.7,
                "tags": ["anger", "threat", "table"],
            },
        ]
    )
    provider = LLMMeaningProvider(llm=llm)
    embedder = HashEmbedding(dim=64, seed=11)
    schema = schema_from_profile(axis_embedder=embedder)
    messages = _multimodal_messages()
    embedding = embedder.embed_content(messages)

    extraction = provider.extract(
        messages,
        embedding,
        schema,
        recent_tags=("conflict",),
    )

    assert extraction.semantic_text == "一个人愤怒地砸桌子，室内对峙明显，视觉上具有压迫感。"
    assert isinstance(extraction.frame.axis_evidence, dict)
    assert extraction.frame.threat == 0.85
    assert extraction.frame.arousal == 0.9
    assert len(llm.calls) == 2

    meaning_call = llm.calls[1]
    messages = meaning_call["messages"]
    assert isinstance(messages, list)
    assert "不能把图片当作装饰" in messages[0].parts[0].text
    assert "如果有图片，要把表情、姿态、距离" in messages[-1].parts[0].text
    assert any(part.type == "image" for message in messages for part in message.parts)
    assert meaning_call["kwargs"]["metadata"]["operation"] == "meaning_extraction_v6"


def test_llm_meaning_provider_project_raises_without_fallback_on_projection_failure() -> None:
    llm = ScriptedLLM([RuntimeError("projection failed")])
    provider = LLMMeaningProvider(llm=llm)

    try:
        provider.project(_multimodal_messages())
    except RuntimeError as exc:
        assert str(exc) == "projection failed"
    else:  # pragma: no cover - defensive
        raise AssertionError("expected RuntimeError")


def test_llm_meaning_provider_extract_raises_without_heuristic_fallback() -> None:
    llm = ScriptedLLM(
        [
            {
                "semantic_text": "一个人愤怒地砸桌子，室内对峙明显，视觉上具有压迫感。",
            },
            RuntimeError("meaning failed"),
        ]
    )
    provider = LLMMeaningProvider(llm=llm)
    embedder = HashEmbedding(dim=64, seed=11)
    schema = schema_from_profile(axis_embedder=embedder)

    try:
        provider.extract(_multimodal_messages(), embedder.embed_content(_multimodal_messages()), schema)
    except RuntimeError as exc:
        assert str(exc) == "meaning failed"
    else:  # pragma: no cover - defensive
        raise AssertionError("expected RuntimeError")


def test_llm_narrative_provider_summary_raises_without_combinatorial_fallback() -> None:
    llm = ScriptedLLM([RuntimeError("summary failed")])
    provider = LLMNarrativeProvider(llm=llm)
    embedder = HashEmbedding(dim=64, seed=11)
    schema = schema_from_profile(axis_embedder=embedder)

    try:
        provider.compose_summary(
            _identity_state(),
            schema,
            recent_semantic_texts=("最近我在反复讨论边界感。",),
        )
    except RuntimeError as exc:
        assert str(exc) == "summary failed"
    else:  # pragma: no cover - defensive
        raise AssertionError("expected RuntimeError")
