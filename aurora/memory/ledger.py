"""Aurora v3 atom recall."""

from __future__ import annotations

import hashlib
import re

import numpy as np
import numpy.typing as npt

from aurora.memory.atoms import active_atoms, atom_text
from aurora.memory.store import SQLiteMemoryStore
from aurora.runtime.contracts import MemoryAtom, RecallHit, RecallResult

EMBEDDING_DIM = 384
FloatVector = npt.NDArray[np.float32]
_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+")
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
_MIN_SCORE = 0.18


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
    """Relation-scoped atom and evidence selector."""

    def __init__(self, store: SQLiteMemoryStore, encoder: HashEmbeddingEncoder | None = None) -> None:
        self.store = store
        self.encoder = encoder or HashEmbeddingEncoder()

    def recall(self, relation_id: str, query: str, limit: int = 5) -> RecallResult:
        if limit <= 0:
            return RecallResult(relation_id=relation_id, query=query, hits=())
        query_vector = self.encoder.encode(query)
        query_tokens = set(tokenize(query))
        hits: list[RecallHit] = []

        for atom in active_atoms(self.store.list_atoms(relation_id)):
            text = atom_text(atom)
            if not text or atom.atom_type in {"forget", "revision", "rule", "lexicon"}:
                continue
            score, why = self._hybrid_score(query_tokens, query_vector, text, self.encoder.encode(text))
            score += self._type_boost(atom)
            score += atom.salience * 0.08
            score *= atom.visibility
            if score < _MIN_SCORE:
                continue
            hits.append(
                RecallHit(
                    item_id=atom.atom_id,
                    kind="atom",
                    content=text,
                    score=score,
                    why_recalled=why,
                    evidence_refs=atom.evidence_event_ids,
                )
            )

        ordered = sorted(hits, key=lambda item: item.score, reverse=True)
        deduped: list[RecallHit] = []
        seen: set[str] = set()
        for hit in ordered:
            if hit.content in seen:
                continue
            seen.add(hit.content)
            deduped.append(hit)
            if len(deduped) >= limit:
                break
        return RecallResult(relation_id=relation_id, query=query, hits=tuple(deduped))

    def _hybrid_score(
        self,
        query_tokens: set[str],
        query_vector: FloatVector,
        content: str,
        content_vector: FloatVector,
    ) -> tuple[float, str]:
        content_tokens = set(tokenize(content))
        lexical = 0.0
        if query_tokens and content_tokens:
            lexical = len(query_tokens & content_tokens) / len(query_tokens | content_tokens)

        vector_score = 0.0
        if float(np.linalg.norm(content_vector)) > 0.0 and float(np.linalg.norm(query_vector)) > 0.0:
            vector_score = float(np.dot(query_vector, content_vector))
            vector_score = max(0.0, vector_score)

        hybrid = lexical * 0.70 + vector_score * 0.30
        if lexical > 0.0 and vector_score > 0.0:
            why = "lexical+vector"
        elif lexical > 0.0:
            why = "lexical"
        elif vector_score > 0.0:
            why = "vector"
        else:
            why = "none"
        return hybrid, why

    def _type_boost(self, atom: MemoryAtom) -> float:
        if atom.atom_type == "loop":
            return 0.10
        if atom.atom_type == "fact":
            return 0.06
        if atom.atom_type == "rule":
            return 0.04
        if atom.atom_type == "lexicon":
            return 0.03
        return 0.0
