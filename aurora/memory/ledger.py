"""Aurora v2 archive recall。"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from aurora.memory.store import SQLiteMemoryStore
from aurora.runtime.contracts import RecallHit, RecallResult

EMBEDDING_DIM = 384
_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+")
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")


def _tokenize(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for chunk in _TOKEN_PATTERN.findall(text.lower()):
        if _CJK_PATTERN.search(chunk):
            tokens.extend(char for char in chunk if _CJK_PATTERN.match(char))
            continue
        tokens.append(chunk)
    return tuple(tokens)


class HashEmbeddingEncoder:
    """无需外部依赖的稳定哈希向量编码器。"""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in _tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm


class Archive:
    """事实与 transcript 的混合检索层。"""

    def __init__(self, store: SQLiteMemoryStore, encoder: HashEmbeddingEncoder | None = None) -> None:
        self.store = store
        self.encoder = encoder or HashEmbeddingEncoder()

    def recall(self, relation_id: str, query: str, limit: int = 5) -> RecallResult:
        if limit <= 0:
            return RecallResult(relation_id=relation_id, query=query, hits=())
        query_vector = self.encoder.encode(query)
        query_tokens = set(_tokenize(query))
        hits: list[RecallHit] = []

        for fact, embedding in self.store.fact_embeddings(relation_id):
            if fact.status not in {"active", "disputed"}:
                continue
            score, why = self._hybrid_score(query_tokens, query_vector, fact.content, embedding)
            if score <= 0.0:
                continue
            hits.append(
                RecallHit(
                    item_id=fact.fact_id,
                    kind="fact",
                    content=fact.content,
                    score=score + 0.05,
                    why_recalled=why,
                    evidence_refs=fact.evidence_refs or (fact.fact_id,),
                )
            )

        for event in self.store.event_rows_for_recall(relation_id):
            event_vector = self.encoder.encode(event.text)
            score, why = self._hybrid_score(query_tokens, query_vector, event.text, event_vector)
            if score <= 0.0:
                continue
            hits.append(
                RecallHit(
                    item_id=event.event_id,
                    kind="event",
                    content=event.text,
                    score=score,
                    why_recalled=why,
                    evidence_refs=(event.event_id,),
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
        query_vector: np.ndarray,
        content: str,
        content_vector: np.ndarray,
    ) -> tuple[float, str]:
        content_tokens = set(_tokenize(content))
        lexical = 0.0
        if query_tokens and content_tokens:
            lexical = len(query_tokens & content_tokens) / len(query_tokens | content_tokens)

        vector_score = 0.0
        if float(np.linalg.norm(content_vector)) > 0.0 and float(np.linalg.norm(query_vector)) > 0.0:
            vector_score = float(np.dot(query_vector, content_vector))
            vector_score = max(0.0, vector_score)

        hybrid = lexical * 0.65 + vector_score * 0.35
        if lexical > 0.0 and vector_score > 0.0:
            why = "lexical+vector"
        elif lexical > 0.0:
            why = "lexical"
        elif vector_score > 0.0:
            why = "vector"
        else:
            why = "none"
        return hybrid, why
