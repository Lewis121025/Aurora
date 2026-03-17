"""Aurora v2 compiler 与 reducer。"""

from __future__ import annotations

import json
from typing import Any, cast
from uuid import uuid4

from aurora.llm.provider import LLMProvider
from aurora.memory.ledger import HashEmbeddingEncoder
from aurora.memory.store import SQLiteMemoryStore
from aurora.relation.state import add_lexicon_items, add_rule, apply_relation_patch
from aurora.runtime.contracts import FactRecord, LoopType, MemoryOp, MemoryOpType, OpenLoop, RelationField, clamp


class CompileError(Exception):
    """编译阶段异常。"""


COMPILER_SYSTEM_PROMPT = """You are Aurora's memory compiler.

Return valid JSON only.
Output either:
{"ops": [...]} or [...]

Allowed op types:
- assert_fact
- revise_fact
- patch_relation
- open_loop
- resolve_loop
- add_rule
- update_lexicon

Rules:
- Only emit ops that should change future behavior.
- Facts must be atomic and specific.
- revise_fact must include target_fact_id from known facts when available.
- patch_relation deltas must stay small and bounded.
- add_rule and update_lexicon must include reason=explicit_feedback or reason=repeated_pattern.
- open_loop is for commitments, contradictions, unfinished threads, and unresolved questions.
- Do not emit duplicates.
"""

_ALLOWED_OP_TYPES = frozenset({
    "assert_fact",
    "revise_fact",
    "patch_relation",
    "open_loop",
    "resolve_loop",
    "add_rule",
    "update_lexicon",
})
_ALLOWED_LOOP_TYPES = frozenset({"commitment", "contradiction", "unfinished_thread", "unresolved_question"})


def compile_memory_ops(
    *,
    pending_turns: tuple[object, ...],
    field: RelationField,
    loops: tuple[OpenLoop, ...],
    facts: tuple[FactRecord, ...],
    llm: LLMProvider,
) -> tuple[MemoryOp, ...]:
    """让 LLM 将 pending turn window 编译为 MemoryOp。"""
    if not pending_turns:
        return ()
    messages = [
        {"role": "system", "content": COMPILER_SYSTEM_PROMPT},
        {"role": "system", "content": _format_state(field, loops, facts)},
        {"role": "user", "content": _format_pending(pending_turns)},
    ]
    raw = llm.complete(messages)
    return parse_memory_ops(raw)


def parse_memory_ops(raw: str) -> tuple[MemoryOp, ...]:
    """解析 compiler 原始 JSON。"""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise CompileError("compiler returned invalid JSON") from exc

    if isinstance(data, dict):
        entries = data.get("ops")
    else:
        entries = data
    if not isinstance(entries, list):
        raise CompileError("compiler output must be a list of ops")

    ops: list[MemoryOp] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise CompileError("compiler op must be an object")
        op_type = entry.get("type") or entry.get("op_type")
        if not isinstance(op_type, str):
            raise CompileError("compiler op missing type")
        if op_type not in _ALLOWED_OP_TYPES:
            raise CompileError(f"unsupported op type: {op_type}")
        payload = entry.get("payload")
        if payload is None:
            payload = {key: value for key, value in entry.items() if key not in {"type", "op_type"}}
        if not isinstance(payload, dict):
            raise CompileError("compiler payload must be an object")
        ops.append(MemoryOp(op_type=cast(MemoryOpType, op_type), payload=payload))
    return tuple(_dedupe_ops(ops))


def apply_memory_ops(
    *,
    store: SQLiteMemoryStore,
    relation_id: str,
    field: RelationField,
    ops: tuple[MemoryOp, ...],
    evidence_refs: tuple[str, ...],
    now_ts: float,
    encoder: HashEmbeddingEncoder,
) -> int:
    """应用 MemoryOp 并保证 reducer 的确定性。"""
    applied = 0
    user_turns = store.user_turn_texts(relation_id)

    for op in ops:
        payload = op.payload
        if op.op_type == "patch_relation":
            apply_relation_patch(
                field,
                trust_delta=float(payload.get("trust_delta", 0.0)),
                distance_delta=float(payload.get("distance_delta", 0.0)),
                warmth_delta=float(payload.get("warmth_delta", 0.0)),
                tension_delta=float(payload.get("tension_delta", 0.0)),
                repair_debt_delta=float(payload.get("repair_debt_delta", 0.0)),
                compiled_at=now_ts,
            )
            applied += 1
            continue

        if op.op_type == "add_rule":
            rule = str(payload.get("rule", "")).strip()
            reason = str(payload.get("reason", ""))
            if _allow_language_update(rule, reason, user_turns):
                add_rule(field, rule)
                applied += 1
            continue

        if op.op_type == "update_lexicon":
            reason = str(payload.get("reason", ""))
            terms_raw = payload.get("terms", [])
            if not isinstance(terms_raw, list):
                raise CompileError("update_lexicon terms must be a list")
            terms = [str(item).strip() for item in terms_raw if str(item).strip()]
            accepted = [term for term in terms if _allow_language_update(term, reason, user_turns)]
            if accepted:
                add_lexicon_items(field, accepted)
                applied += 1
            continue

        if op.op_type == "open_loop":
            loop_type = str(payload.get("loop_type", "")).strip()
            summary = str(payload.get("summary", "")).strip()
            urgency = clamp(float(payload.get("urgency", 0.5)))
            if not loop_type or not summary:
                raise CompileError("open_loop requires loop_type and summary")
            if loop_type not in _ALLOWED_LOOP_TYPES:
                raise CompileError(f"open_loop received invalid loop_type: {loop_type}")
            existing = store.find_open_loop(relation_id, loop_type, summary)
            loop_id = existing.loop_id if existing is not None else f"loop_{uuid4().hex[:12]}"
            opened_at = existing.opened_at if existing is not None else now_ts
            refs = evidence_refs if existing is None else tuple(dict.fromkeys((*existing.evidence_refs, *evidence_refs)))
            store.upsert_open_loop(
                OpenLoop(
                    loop_id=loop_id,
                    relation_id=relation_id,
                    loop_type=cast(LoopType, loop_type),
                    status="active",
                    summary=summary,
                    urgency=max(urgency, existing.urgency) if existing is not None else urgency,
                    opened_at=opened_at,
                    updated_at=now_ts,
                    evidence_refs=refs,
                )
            )
            applied += 1
            continue

        if op.op_type == "resolve_loop":
            resolved = store.resolve_open_loop(
                relation_id,
                loop_id=_optional_str(payload.get("loop_id")),
                summary=_optional_str(payload.get("summary")),
                loop_type=_optional_str(payload.get("loop_type")),
                updated_at=now_ts,
                evidence_refs=evidence_refs,
            )
            if resolved:
                applied += 1
            continue

        if op.op_type == "assert_fact":
            content = str(payload.get("content", "")).strip()
            if not content:
                raise CompileError("assert_fact requires content")
            if any(fact.content == content and fact.status == "active" for fact in store.list_facts(relation_id)):
                continue
            record = FactRecord(
                fact_id=f"fact_{uuid4().hex[:12]}",
                relation_id=relation_id,
                content=content,
                document_date=now_ts,
                event_date=float(payload.get("event_date", now_ts)),
                status="active",
                supersedes=None,
                confidence=clamp(float(payload.get("confidence", 0.7))),
                evidence_refs=evidence_refs,
            )
            store.add_fact(record, encoder.encode(content))
            applied += 1
            continue

        if op.op_type == "revise_fact":
            content = str(payload.get("content", "")).strip()
            target_fact_id = _optional_str(payload.get("target_fact_id"))
            if not content or target_fact_id is None:
                raise CompileError("revise_fact requires content and target_fact_id")
            target = store.get_fact(target_fact_id)
            if target is None:
                raise CompileError("revise_fact target_fact_id does not exist")
            if target.content == content:
                continue
            store.update_fact_status(target.fact_id, "superseded")
            revised = FactRecord(
                fact_id=f"fact_{uuid4().hex[:12]}",
                relation_id=relation_id,
                content=content,
                document_date=now_ts,
                event_date=float(payload.get("event_date", now_ts)),
                status="active",
                supersedes=target.fact_id,
                confidence=clamp(float(payload.get("confidence", target.confidence))),
                evidence_refs=evidence_refs,
            )
            store.add_fact(revised, encoder.encode(content))
            summary = payload.get("summary") or f"Fact contradiction: {target.content} -> {content}"
            store.upsert_open_loop(
                OpenLoop(
                    loop_id=f"loop_{uuid4().hex[:12]}",
                    relation_id=relation_id,
                    loop_type="contradiction",
                    status="active",
                    summary=str(summary),
                    urgency=clamp(float(payload.get("urgency", 0.75))),
                    opened_at=now_ts,
                    updated_at=now_ts,
                    evidence_refs=tuple(dict.fromkeys((*target.evidence_refs, *evidence_refs, revised.fact_id))),
                )
            )
            applied += 1
            continue

        raise CompileError(f"unsupported op type: {op.op_type}")

    field.last_compiled_at = now_ts
    return applied


def _format_state(field: RelationField, loops: tuple[OpenLoop, ...], facts: tuple[FactRecord, ...]) -> str:
    lines = [
        "Current relation field:",
        f"- trust={field.trust:.2f}",
        f"- distance={field.distance:.2f}",
        f"- warmth={field.warmth:.2f}",
        f"- tension={field.tension:.2f}",
        f"- repair_debt={field.repair_debt:.2f}",
        f"- rules={field.interaction_rules}",
        f"- lexicon={field.shared_lexicon}",
        "",
        "Active open loops:",
    ]
    if loops:
        lines.extend(f"- {loop.loop_id}: {loop.loop_type} :: {loop.summary}" for loop in loops)
    else:
        lines.append("- none")
    lines.append("")
    lines.append("Known facts:")
    if facts:
        lines.extend(
            f"- {fact.fact_id}: {fact.content} [{fact.status}]"
            for fact in facts[:20]
        )
    else:
        lines.append("- none")
    return "\n".join(lines)


def _format_pending(pending_turns: tuple[object, ...]) -> str:
    lines = ["Pending conversation window:"]
    for event in pending_turns:
        role = getattr(event, "role")
        text = getattr(event, "text")
        lines.append(f"- {role}: {text}")
    return "\n".join(lines)


def _allow_language_update(text: str, reason: str, user_turns: tuple[str, ...]) -> bool:
    if not text:
        return False
    if reason == "explicit_feedback":
        markers = ("不要", "别", "请", "记住", "never", "don't", "please")
        return any(marker in turn for turn in user_turns for marker in markers)
    if reason == "repeated_pattern":
        return sum(1 for turn in user_turns if text in turn) >= 2
    return False


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _dedupe_ops(ops: list[MemoryOp]) -> list[MemoryOp]:
    deduped: list[MemoryOp] = []
    seen: set[str] = set()
    for op in ops:
        marker = json.dumps({"type": op.op_type, "payload": op.payload}, sort_keys=True, ensure_ascii=False)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(op)
    return deduped
