from __future__ import annotations

from typing import Any, Sequence

from pydantic import TypeAdapter

from aurora.interfaces.api.schemas import MessagePayload
from aurora.soul.models import ImagePart, Message, TextPart

MESSAGE_PAYLOADS = TypeAdapter(list[MessagePayload])


def to_domain_messages(items: Sequence[MessagePayload]) -> list[Message]:
    messages: list[Message] = []
    for item in items:
        parts = []
        for part in item.parts:
            if part.type == "text":
                parts.append(TextPart(text=part.text))
            else:
                parts.append(ImagePart(uri=part.uri, mime_type=part.mime_type))
        messages.append(Message(role=item.role, parts=tuple(parts), actor=item.actor))
    return messages


def parse_message_json(raw: str) -> list[Message]:
    payloads = MESSAGE_PAYLOADS.validate_json(raw)
    return to_domain_messages(payloads)


def parse_message_payloads(raw: Any) -> list[Message]:
    payloads = MESSAGE_PAYLOADS.validate_python(raw)
    return to_domain_messages(payloads)


def message_payload(message: Message) -> MessagePayload:
    return MessagePayload.model_validate(message.to_state_dict())
