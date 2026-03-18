"""Subject-scoped memory recall."""

from __future__ import annotations

import hashlib
import re
from typing import cast

import numpy as np
import numpy.typing as npt

from aurora.memory.state import atom_text, suppressed_source_ids, visible_recall_atoms
from aurora.memory.store import SQLiteMemoryStore
from aurora.runtime.contracts import MemoryAtom, MemoryKind, RecallHit, RecallMode, RecallResult, RecallTemporalScope

EMBEDDING_DIM = 384
FloatVector = npt.NDArray[np.float32]
_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+")
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
_BLENDED_MEMORY_KINDS: tuple[MemoryKind, ...] = (
    "episode",
    "semantic",
    "procedural",
    "cognitive",
    "affective",
    "narrative",
)


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize mixed CJK and latin text."""
    tokens: list[str] = []
    for chunk in _TOKEN_PATTERN.findall(text.lower()):
        if _CJK_PATTERN.search(chunk):
            tokens.extend(char for char in chunk if _CJK_PATTERN.match(char))
            continue
        tokens.append(chunk)
    return tuple(tokens)


class HashEmbeddingEncoder:
    """Stable hash encoder without external dependencies."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    def encode(self, text: str) -> FloatVector:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm


class Archive:
    """Selector over recallable subject memory."""

    def __init__(self, store: SQLiteMemoryStore, encoder: HashEmbeddingEncoder | None = None) -> None:
        self.store = store
        self.encoder = encoder or HashEmbeddingEncoder()

    def recall(
        self,
        subject_id: str,
        query: str,
        *,
        temporal_scope: RecallTemporalScope,
        limit: int = 5,
        mode: RecallMode = "blended",
    ) -> RecallResult:
        if limit <= 0:
            return RecallResult(subject_id=subject_id, query=query, temporal_scope=temporal_scope, mode=mode, hits=())

        atoms = self.store.list_atoms(subject_id)
        query_tokens = set(tokenize(query))
        query_vector = self.encoder.encode(query)
        blocked_sources = set(suppressed_source_ids(atoms)) if temporal_scope == "current" else set()
        ranked_hits: list[tuple[tuple[int, float, float, float, float, float], RecallHit]] = []

        for atom in visible_recall_atoms(atoms, include_superseded=temporal_scope != "current"):
            if not _matches_mode(atom, mode):
                continue
            if not _matches_temporal_scope(atom, temporal_scope):
                continue
            if blocked_sources and atom.atom_kind in {"episode", "narrative"}:
                if any(source_id in blocked_sources for source_id in atom.source_atom_ids):
                    continue
            text = atom_text(atom)
            if not text:
                continue
            lexical, vector_score = _relevance(query_tokens, query_vector, text, self.encoder)
            if lexical <= 0.0 and vector_score <= 0.0:
                continue
            ranked_hits.append(
                (
                    _sort_key(
                        atom=atom,
                        lexical=lexical,
                        vector_score=vector_score,
                        temporal_scope=temporal_scope,
                    ),
                    RecallHit(
                        memory_kind=cast(MemoryKind, atom.atom_kind),
                        content=text,
                        score=max(lexical, vector_score),
                        why_recalled=_why_recalled(
                            atom=atom,
                            lexical=lexical,
                            vector_score=vector_score,
                            temporal_scope=temporal_scope,
                        ),
                    ),
                )
            )

        ordered = _dedupe_hits(
            tuple(hit for _, hit in sorted(ranked_hits, key=lambda item: item[0], reverse=True))
        )
        return RecallResult(
            subject_id=subject_id,
            query=query,
            temporal_scope=temporal_scope,
            mode=mode,
            hits=ordered[:limit],
        )


def _relevance(
    query_tokens: set[str],
    query_vector: FloatVector,
    content: str,
    encoder: HashEmbeddingEncoder,
) -> tuple[float, float]:
    content_tokens = set(tokenize(content))
    lexical = 0.0
    if query_tokens and content_tokens:
        lexical = len(query_tokens & content_tokens) / len(query_tokens | content_tokens)

    content_vector = encoder.encode(content)
    vector_score = 0.0
    if float(np.linalg.norm(content_vector)) > 0.0 and float(np.linalg.norm(query_vector)) > 0.0:
        vector_score = max(0.0, float(np.dot(query_vector, content_vector)))
    return lexical, vector_score


def _matches_mode(atom: MemoryAtom, mode: RecallMode) -> bool:
    if mode == "blended":
        return atom.atom_kind in _BLENDED_MEMORY_KINDS
    if mode == "episodic":
        return atom.atom_kind == "episode"
    return atom.atom_kind == mode


def _matches_temporal_scope(atom: MemoryAtom, temporal_scope: RecallTemporalScope) -> bool:
    if atom.status == "inhibited":
        return False
    if temporal_scope == "current":
        return atom.status == "active"
    if temporal_scope == "historical":
        return atom.status == "superseded"
    return atom.status in {"active", "superseded"}


def _sort_key(
    *,
    atom: MemoryAtom,
    lexical: float,
    vector_score: float,
    temporal_scope: RecallTemporalScope,
) -> tuple[int, float, float, float, float, float]:
    return (
        _status_priority(atom.status, temporal_scope),
        lexical,
        vector_score,
        atom.salience,
        atom.accessibility,
        atom.updated_at,
    )


def _status_priority(status: str, temporal_scope: RecallTemporalScope) -> int:
    if temporal_scope == "current":
        return 1 if status == "active" else 0
    if temporal_scope == "historical":
        return 1 if status == "superseded" else 0
    return 0


def _why_recalled(
    atom: MemoryAtom,
    *,
    lexical: float,
    vector_score: float,
    temporal_scope: RecallTemporalScope,
) -> str:
    signals: list[str] = []
    if lexical > 0.0:
        signals.append("lexical")
    if vector_score > 0.0:
        signals.append("vector")
    if temporal_scope == "current" and atom.status == "active":
        signals.append("current")
    if temporal_scope == "historical" and atom.status == "superseded":
        signals.append("historical")
    return "+".join(signals) or "matched"


def _dedupe_hits(hits: tuple[RecallHit, ...]) -> tuple[RecallHit, ...]:
    seen: set[tuple[MemoryKind, str]] = set()
    unique: list[RecallHit] = []
    for hit in hits:
        key = (hit.memory_kind, hit.content)
        if key in seen:
            continue
        seen.add(key)
        unique.append(hit)
    return tuple(unique)
