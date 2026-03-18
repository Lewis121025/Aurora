"""Aurora v3 compiler and atom reducer."""

from __future__ import annotations

import json
import re
from typing import Any, cast
from uuid import uuid4

from aurora.llm.provider import LLMProvider
from aurora.memory.atoms import active_atoms, atom_text, effective_atoms
from aurora.memory.ledger import tokenize
from aurora.memory.store import SQLiteMemoryStore
from aurora.runtime.contracts import MemoryAtom, MemoryOp, MemoryOpType, clamp


class CompileError(Exception):
    """Compiler stage error."""


POST_RESPONSE_COMPILER_SYSTEM_PROMPT = """You are Aurora's memory compiler.

Return valid JSON only.
Output either:
{"ops": [...]} or [...]

Allowed atom op types:
- fact
- rule
- lexicon
- loop
- revision
- forget

Rules:
- Only emit atoms that should change future continuity.
- fact requires content and fact_kind.
- revision requires target_atom_id and content.
- loop requires loop_type and summary. Use state=resolved when closing one.
- forget is only for explicit user intent to stop carrying something forward.
- Keep outputs compact and non-duplicated.
"""

_ALLOWED_OP_TYPES = frozenset({"fact", "rule", "lexicon", "loop", "revision", "forget"})
_ALLOWED_LOOP_TYPES = frozenset({"commitment", "contradiction", "unfinished_thread", "unresolved_question"})
_FORGET_PATTERNS = (
    "别再记",
    "不要再记",
    "别记",
    "不要记",
    "忘了这个",
    "forget this",
    "do not remember",
)
_COMMITMENT_PATTERNS = ("提醒我", "记得提醒", "之后提醒", "下次提醒")
_UNFINISHED_PATTERNS = ("之后再聊", "下次再说", "回头继续", "先放这", "以后继续")
_FACT_KIND_RE = re.compile(r"(喜欢|偏好|想要)")
_RULE_DIRECTIVE_MARKERS = ("别", "不要", "请", "direct", "直接", "保持")
_PERSISTENT_RULE_MARKERS = ("以后", "今后", "下次", "之后", "别再", "不要再", "记住", "一直", "长期")


def compile_pre_response_ops(
    *,
    text: str,
    atoms: tuple[MemoryAtom, ...],
) -> tuple[MemoryOp, ...]:
    """Deterministically absorb explicit high-value signals before response."""
    stripped = text.strip()
    if not stripped:
        return ()

    ops: list[MemoryOp] = []
    if any(marker in stripped for marker in _FORGET_PATTERNS):
        ops.append(
            MemoryOp(
                op_type="forget",
                payload={
                    "matcher": stripped,
                    "target_atom_ids": list(_match_related_atoms(stripped, atoms)),
                    "summary": stripped,
                },
            )
        )

    if _should_persist_rule(stripped, "explicit_feedback"):
        ops.append(
            MemoryOp(
                op_type="rule",
                payload={"text": stripped, "reason": "explicit_feedback"},
            )
        )

    if any(marker in stripped for marker in _COMMITMENT_PATTERNS):
        ops.append(
            MemoryOp(
                op_type="loop",
                payload={
                    "state": "active",
                    "loop_type": "commitment",
                    "summary": stripped,
                    "urgency": 0.78,
                },
            )
        )
    elif any(marker in stripped for marker in _UNFINISHED_PATTERNS):
        ops.append(
            MemoryOp(
                op_type="loop",
                payload={
                    "state": "active",
                    "loop_type": "unfinished_thread",
                    "summary": stripped,
                    "urgency": 0.66,
                },
            )
        )

    return tuple(_dedupe_ops(ops))


def compile_post_response_ops(
    *,
    relation_id: str,
    user_turn: str,
    assistant_turn: str,
    atoms: tuple[MemoryAtom, ...],
    llm: LLMProvider,
) -> tuple[MemoryOp, ...]:
    """Compile one completed turn into memory atom operations."""
    messages = [
        {"role": "system", "content": POST_RESPONSE_COMPILER_SYSTEM_PROMPT},
        {"role": "system", "content": _format_state(atoms)},
        {"role": "user", "content": _format_turn(relation_id, user_turn, assistant_turn)},
    ]
    raw = llm.complete(messages)
    return parse_memory_ops(raw)


def parse_memory_ops(raw: str) -> tuple[MemoryOp, ...]:
    """Parse compiler JSON."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise CompileError("compiler returned invalid JSON") from exc

    entries = data.get("ops") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise CompileError("compiler output must be a list of ops")

    ops: list[MemoryOp] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise CompileError("compiler op must be an object")
        op_type = entry.get("type") or entry.get("op_type")
        if not isinstance(op_type, str) or op_type not in _ALLOWED_OP_TYPES:
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
    ops: tuple[MemoryOp, ...],
    evidence_event_ids: tuple[str, ...],
    now_ts: float,
) -> tuple[str, ...]:
    """Apply compiled ops as memory atoms and lifecycle updates."""
    applied: list[str] = []
    atoms = store.list_atoms(relation_id)

    for op in ops:
        if op.op_type == "fact":
            atom_id = _apply_fact(store, relation_id, op, evidence_event_ids, now_ts, atoms)
            if atom_id is not None:
                applied.append(atom_id)
                atoms = store.list_atoms(relation_id)
            continue

        if op.op_type == "rule":
            atom_id = _apply_rule(store, relation_id, op, evidence_event_ids, now_ts, atoms)
            if atom_id is not None:
                applied.append(atom_id)
                atoms = store.list_atoms(relation_id)
            continue

        if op.op_type == "lexicon":
            ids = _apply_lexicon(store, relation_id, op, evidence_event_ids, now_ts, atoms)
            if ids:
                applied.extend(ids)
                atoms = store.list_atoms(relation_id)
            continue

        if op.op_type == "loop":
            atom_id = _apply_loop(store, relation_id, op, evidence_event_ids, now_ts, atoms)
            if atom_id is not None:
                applied.append(atom_id)
                atoms = store.list_atoms(relation_id)
            continue

        if op.op_type == "revision":
            ids = _apply_revision(store, relation_id, op, evidence_event_ids, now_ts)
            if ids:
                applied.extend(ids)
                atoms = store.list_atoms(relation_id)
            continue

        if op.op_type == "forget":
            ids = _apply_forget(store, relation_id, op, evidence_event_ids, now_ts, atoms)
            if ids:
                applied.extend(ids)
                atoms = store.list_atoms(relation_id)
            continue

        raise CompileError(f"unsupported op type: {op.op_type}")

    return tuple(applied)


def _apply_fact(
    store: SQLiteMemoryStore,
    relation_id: str,
    op: MemoryOp,
    evidence_event_ids: tuple[str, ...],
    now_ts: float,
    atoms: tuple[MemoryAtom, ...],
) -> str | None:
    content = str(op.payload.get("content", "")).strip()
    if not content:
        raise CompileError("fact requires content")
    fact_kind = _fact_kind(op.payload.get("fact_kind"))
    payload = {
        "content": content,
        "fact_kind": fact_kind,
        "event_date": float(op.payload.get("event_date", now_ts)),
    }
    if _find_effective_equivalent_atom(atoms, "fact", payload) is not None:
        return None
    atom = MemoryAtom(
        atom_id=f"atom_{uuid4().hex[:12]}",
        relation_id=relation_id,
        atom_type="fact",
        payload=payload,
        status="active",
        confidence=clamp(float(op.payload.get("confidence", 0.78))),
        salience=clamp(float(op.payload.get("salience", 0.72))),
        visibility=1.0,
        evidence_event_ids=evidence_event_ids,
        created_at=now_ts,
        updated_at=now_ts,
    )
    store.add_atom(atom)
    return atom.atom_id


def _apply_rule(
    store: SQLiteMemoryStore,
    relation_id: str,
    op: MemoryOp,
    evidence_event_ids: tuple[str, ...],
    now_ts: float,
    atoms: tuple[MemoryAtom, ...],
) -> str | None:
    text = str(op.payload.get("text") or op.payload.get("rule") or "").strip()
    if not text:
        raise CompileError("rule requires text")
    reason = str(op.payload.get("reason", ""))
    if not _should_persist_rule(text, reason):
        return None
    payload = {"text": text, "reason": reason}
    if _find_effective_equivalent_atom(atoms, "rule", payload) is not None:
        return None
    atom = MemoryAtom(
        atom_id=f"atom_{uuid4().hex[:12]}",
        relation_id=relation_id,
        atom_type="rule",
        payload=payload,
        status="active",
        confidence=clamp(float(op.payload.get("confidence", 0.90))),
        salience=clamp(float(op.payload.get("salience", 0.85))),
        visibility=1.0,
        evidence_event_ids=evidence_event_ids,
        created_at=now_ts,
        updated_at=now_ts,
    )
    store.add_atom(atom)
    return atom.atom_id


def _apply_lexicon(
    store: SQLiteMemoryStore,
    relation_id: str,
    op: MemoryOp,
    evidence_event_ids: tuple[str, ...],
    now_ts: float,
    atoms: tuple[MemoryAtom, ...],
) -> tuple[str, ...]:
    terms_raw = op.payload.get("terms")
    if not isinstance(terms_raw, list):
        raise CompileError("lexicon terms must be a list")
    created: list[str] = []
    for term in (str(item).strip() for item in terms_raw):
        if not term:
            continue
        payload = {"terms": [term], "reason": str(op.payload.get("reason", ""))}
        if _find_effective_equivalent_atom(atoms, "lexicon", payload) is not None:
            continue
        atom = MemoryAtom(
            atom_id=f"atom_{uuid4().hex[:12]}",
            relation_id=relation_id,
            atom_type="lexicon",
            payload=payload,
            status="active",
            confidence=clamp(float(op.payload.get("confidence", 0.75))),
            salience=clamp(float(op.payload.get("salience", 0.60))),
            visibility=1.0,
            evidence_event_ids=evidence_event_ids,
            created_at=now_ts,
            updated_at=now_ts,
        )
        store.add_atom(atom)
        created.append(atom.atom_id)
    return tuple(created)


def _apply_loop(
    store: SQLiteMemoryStore,
    relation_id: str,
    op: MemoryOp,
    evidence_event_ids: tuple[str, ...],
    now_ts: float,
    atoms: tuple[MemoryAtom, ...],
) -> str | None:
    loop_type = str(op.payload.get("loop_type", "")).strip()
    summary = str(op.payload.get("summary", "")).strip()
    if not loop_type or not summary:
        raise CompileError("loop requires loop_type and summary")
    if loop_type not in _ALLOWED_LOOP_TYPES:
        raise CompileError(f"loop received invalid loop_type: {loop_type}")

    state = str(op.payload.get("state", "active")).strip().lower()
    if state == "resolved":
        loop_ref = _resolve_loop_ref(op.payload, atoms)
        if loop_ref is None:
            return None
        payload = {
            "loop_ref": loop_ref,
            "loop_type": loop_type,
            "summary": summary,
            "urgency": clamp(float(op.payload.get("urgency", 0.5))),
            "state": "resolved",
            "opened_at": float(op.payload.get("opened_at", now_ts)),
        }
    else:
        loop_ref = str(op.payload.get("loop_ref") or f"loop_{uuid4().hex[:12]}")
        payload = {
            "loop_ref": loop_ref,
            "loop_type": loop_type,
            "summary": summary,
            "urgency": clamp(float(op.payload.get("urgency", 0.5))),
            "state": "active",
            "opened_at": float(op.payload.get("opened_at", now_ts)),
        }
        if _find_effective_equivalent_atom(atoms, "loop", payload) is not None:
            return None

    atom = MemoryAtom(
        atom_id=f"atom_{uuid4().hex[:12]}",
        relation_id=relation_id,
        atom_type="loop",
        payload=payload,
        status="resolved" if state == "resolved" else "active",
        confidence=clamp(float(op.payload.get("confidence", 0.80))),
        salience=clamp(float(op.payload.get("salience", payload["urgency"]))),
        visibility=1.0,
        evidence_event_ids=evidence_event_ids,
        affects_atom_ids=(loop_ref,),
        created_at=now_ts,
        updated_at=now_ts,
    )
    store.add_atom(atom)
    return atom.atom_id


def _apply_revision(
    store: SQLiteMemoryStore,
    relation_id: str,
    op: MemoryOp,
    evidence_event_ids: tuple[str, ...],
    now_ts: float,
) -> tuple[str, ...]:
    content = str(op.payload.get("content", "")).strip()
    target_atom_id = _optional_str(op.payload.get("target_atom_id"))
    if not content or target_atom_id is None:
        raise CompileError("revision requires target_atom_id and content")
    target = store.get_atom(target_atom_id)
    if target is None or target.atom_type != "fact":
        raise CompileError("revision target_atom_id does not exist")
    if target.relation_id != relation_id:
        raise CompileError("revision target_atom_id belongs to another relation")
    if atom_text(target) == content:
        return ()

    fact_kind = _fact_kind(op.payload.get("fact_kind") or target.payload.get("fact_kind"))
    revised_fact = MemoryAtom(
        atom_id=f"atom_{uuid4().hex[:12]}",
        relation_id=relation_id,
        atom_type="fact",
        payload={
            "content": content,
            "fact_kind": fact_kind,
            "event_date": float(op.payload.get("event_date", now_ts)),
        },
        status="active",
        confidence=clamp(float(op.payload.get("confidence", target.confidence))),
        salience=clamp(float(op.payload.get("salience", max(target.salience, 0.78)))),
        visibility=1.0,
        evidence_event_ids=evidence_event_ids,
        supersedes_atom_id=target.atom_id,
        created_at=now_ts,
        updated_at=now_ts,
    )
    store.add_atom(revised_fact)

    revision_atom = MemoryAtom(
        atom_id=f"atom_{uuid4().hex[:12]}",
        relation_id=relation_id,
        atom_type="revision",
        payload={
            "target_atom_id": target.atom_id,
            "content": content,
            "summary": str(op.payload.get("summary", "")).strip(),
        },
        status="active",
        confidence=clamp(float(op.payload.get("confidence", target.confidence))),
        salience=clamp(float(op.payload.get("salience", 0.82))),
        visibility=0.4,
        evidence_event_ids=evidence_event_ids,
        affects_atom_ids=(target.atom_id, revised_fact.atom_id),
        supersedes_atom_id=target.atom_id,
        created_at=now_ts,
        updated_at=now_ts,
    )
    store.add_atom(revision_atom)
    return (revised_fact.atom_id, revision_atom.atom_id)


def _apply_forget(
    store: SQLiteMemoryStore,
    relation_id: str,
    op: MemoryOp,
    evidence_event_ids: tuple[str, ...],
    now_ts: float,
    atoms: tuple[MemoryAtom, ...],
) -> tuple[str, ...]:
    matcher = str(op.payload.get("matcher", "")).strip()
    explicit_targets = op.payload.get("target_atom_ids", [])
    target_atom_ids = [str(item) for item in explicit_targets] if isinstance(explicit_targets, list) else []
    if not target_atom_ids:
        target_atom_ids = list(_match_related_atoms(matcher, atoms))
    if not target_atom_ids:
        return ()

    related = _locally_related_atoms(target_atom_ids, atoms)

    forget_atom = MemoryAtom(
        atom_id=f"atom_{uuid4().hex[:12]}",
        relation_id=relation_id,
        atom_type="forget",
        payload={
            "matcher": matcher,
            "summary": str(op.payload.get("summary", matcher)).strip(),
        },
        status="active",
        confidence=clamp(float(op.payload.get("confidence", 0.92))),
        salience=clamp(float(op.payload.get("salience", 0.88))),
        visibility=0.45,
        evidence_event_ids=evidence_event_ids,
        affects_atom_ids=tuple(dict.fromkeys([*target_atom_ids, *related])),
        created_at=now_ts,
        updated_at=now_ts,
    )
    store.add_atom(forget_atom)
    return (forget_atom.atom_id,)


def _format_state(atoms: tuple[MemoryAtom, ...]) -> str:
    lines = ["Current visible atoms:"]
    visible = [
        atom
        for atom in active_atoms(atoms)
        if atom.atom_type != "forget"
    ]
    if not visible:
        lines.append("- none")
        return "\n".join(lines)
    for atom in visible[:24]:
        lines.append(
            f"- {atom.atom_id}: {atom.atom_type} :: {atom_text(atom)} "
            f"[status={atom.status}, visibility={atom.visibility:.2f}]"
        )
    return "\n".join(lines)


def _format_turn(relation_id: str, user_turn: str, assistant_turn: str) -> str:
    return (
        f"Relation: {relation_id}\n"
        "Completed turn:\n"
        f"- user: {user_turn}\n"
        f"- assistant: {assistant_turn}"
    )


def _fact_kind(value: object) -> str:
    if value in {"profile", "preference", "current_state", "biographical"}:
        return str(value)
    return "preference" if _FACT_KIND_RE.search(str(value or "")) else "profile"


def _should_persist_rule(text: str, reason: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    if reason == "repeated_pattern":
        return True
    if "记住" in normalized:
        return True
    has_directive = any(marker in normalized for marker in _RULE_DIRECTIVE_MARKERS)
    has_persistent_scope = any(marker in normalized for marker in _PERSISTENT_RULE_MARKERS)
    return has_directive and has_persistent_scope


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _resolve_loop_ref(payload: dict[str, Any], atoms: tuple[MemoryAtom, ...]) -> str | None:
    target_atom_id = _optional_str(payload.get("target_atom_id"))
    if target_atom_id is not None:
        target = next((atom for atom in atoms if atom.atom_id == target_atom_id), None)
        if target is not None:
            return str(target.payload.get("loop_ref") or target.atom_id)
        return None
    loop_ref = _optional_str(payload.get("loop_ref"))
    if loop_ref is not None:
        return loop_ref
    summary = str(payload.get("summary", "")).strip()
    loop_type = str(payload.get("loop_type", "")).strip()
    for atom in atoms:
        if atom.atom_type != "loop":
            continue
        if str(atom.payload.get("summary", "")).strip() == summary and str(atom.payload.get("loop_type", "")) == loop_type:
            return str(atom.payload.get("loop_ref") or atom.atom_id)
    return None


def _match_related_atoms(text: str, atoms: tuple[MemoryAtom, ...]) -> tuple[str, ...]:
    query_tokens = set(tokenize(text))
    if not query_tokens:
        return ()
    ranked: list[tuple[float, str]] = []
    for atom in active_atoms(atoms):
        if atom.atom_type == "forget":
            continue
        content = atom_text(atom)
        if not content:
            continue
        tokens = set(tokenize(content))
        if not tokens:
            continue
        score = len(tokens & query_tokens) / len(tokens | query_tokens)
        if score <= 0.0:
            continue
        ranked.append((score, atom.atom_id))
    ranked.sort(reverse=True)
    return tuple(atom_id for _, atom_id in ranked[:4])


def _locally_related_atoms(target_atom_ids: list[str], atoms: tuple[MemoryAtom, ...]) -> tuple[str, ...]:
    effective = effective_atoms(atoms)
    targets = {atom.atom_id: atom for atom in effective if atom.atom_id in set(target_atom_ids)}
    if not targets:
        return ()
    related: list[str] = []
    target_tokens = {token for atom in targets.values() for token in tokenize(atom_text(atom))}
    target_kinds = {atom.atom_type for atom in targets.values()}
    for atom in active_atoms(effective):
        if atom.atom_id in targets:
            continue
        if atom.atom_type not in target_kinds:
            continue
        content_tokens = set(tokenize(atom_text(atom)))
        if not content_tokens:
            continue
        overlap = len(content_tokens & target_tokens) / len(content_tokens | target_tokens)
        if overlap >= 0.45:
            related.append(atom.atom_id)
    return tuple(related[:6])


def _find_effective_equivalent_atom(
    atoms: tuple[MemoryAtom, ...],
    atom_type: str,
    payload: dict[str, Any],
) -> MemoryAtom | None:
    marker = _payload_signature(payload)
    for atom in effective_atoms(atoms):
        if atom.atom_type != atom_type or atom.status != "active" or atom.visibility <= 0.0:
            continue
        if _payload_signature(atom.payload) == marker:
            return atom
    return None


def _payload_signature(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


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
