from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from aurora.memory.store import SQLiteMemoryStore
from aurora.runtime.contracts import EvidenceContent, MemoryAtom, MemoryContent, MemoryEdge, atom_kind_from_value


@pytest.fixture
def store(tmp_path: Path) -> Iterator[SQLiteMemoryStore]:
    memory_store = SQLiteMemoryStore(str(tmp_path / "aurora.db"))
    yield memory_store
    memory_store.close()


def _memory_atom(
    *,
    atom_id: str,
    subject_id: str = "subject-a",
    text: str = "valid memory",
    confidence: float = 0.9,
    salience: float = 0.8,
) -> MemoryAtom:
    return MemoryAtom(
        atom_id=atom_id,
        subject_id=subject_id,
        atom_kind="memory",
        content=MemoryContent(text=text),
        confidence=confidence,
        salience=salience,
        created_at=1.0,
    )


def _edge(
    *,
    edge_id: str,
    source_atom_id: str,
    target_atom_id: str,
    subject_id: str = "subject-a",
    influence: float = 0.8,
    confidence: float = 0.9,
) -> MemoryEdge:
    return MemoryEdge(
        edge_id=edge_id,
        subject_id=subject_id,
        source_atom_id=source_atom_id,
        target_atom_id=target_atom_id,
        influence=influence,
        confidence=confidence,
        created_at=2.0,
    )


def _evidence_atom(
    *,
    atom_id: str,
    subject_id: str = "subject-a",
    text: str = "observed trace",
    source_atom_ids: tuple[str, ...] = (),
) -> MemoryAtom:
    return MemoryAtom(
        atom_id=atom_id,
        subject_id=subject_id,
        atom_kind="evidence",
        content=EvidenceContent(
            event_kind="user_turn",
            role="user",
            text=text,
            payload={},
        ),
        confidence=1.0,
        salience=0.2,
        source_atom_ids=source_atom_ids,
        created_at=1.0,
    )


def test_store_axiom_rejects_invalid_atom_kind_and_score_ranges(store: SQLiteMemoryStore) -> None:
    with pytest.raises(ValueError, match="invalid atom_kind"):
        atom_kind_from_value("unknown")

    with pytest.raises(sqlite3.IntegrityError):
        store.add_atom(
            MemoryAtom(
                atom_id="atom-invalid-kind",
                subject_id="subject-a",
                atom_kind=cast(Any, "unknown"),
                content=MemoryContent(text="invalid atom kind"),
                confidence=0.9,
                salience=0.8,
                created_at=1.0,
            )
        )

    with pytest.raises(sqlite3.IntegrityError):
        store.add_atom(_memory_atom(atom_id="atom-invalid-confidence", confidence=1.2))

    with pytest.raises(sqlite3.IntegrityError):
        store.add_atom(_memory_atom(atom_id="atom-invalid-salience", salience=-0.1))

    with pytest.raises(sqlite3.IntegrityError):
        store.add_atom(_evidence_atom(atom_id="atom-invalid-evidence", source_atom_ids=("atom-source",)))


def test_store_axiom_rejects_cross_subject_unknown_and_self_loop_edges(store: SQLiteMemoryStore) -> None:
    source = _memory_atom(atom_id="atom-source", subject_id="subject-a")
    foreign = _memory_atom(atom_id="atom-foreign", subject_id="subject-b")
    store.add_atom(source)
    store.add_atom(foreign)

    with pytest.raises(sqlite3.IntegrityError):
        store.add_edge(
            _edge(
                edge_id="edge-cross-subject",
                subject_id="subject-a",
                source_atom_id=source.atom_id,
                target_atom_id=foreign.atom_id,
            )
        )

    with pytest.raises(sqlite3.IntegrityError):
        store.add_edge(
            _edge(
                edge_id="edge-unknown-target",
                source_atom_id=source.atom_id,
                target_atom_id="missing-target",
            )
        )

    with pytest.raises(sqlite3.IntegrityError):
        store.add_edge(
            _edge(
                edge_id="edge-self-loop",
                source_atom_id=source.atom_id,
                target_atom_id=source.atom_id,
            )
        )


def test_store_axiom_rejects_edges_touching_evidence_atoms(store: SQLiteMemoryStore) -> None:
    evidence = _evidence_atom(atom_id="atom-evidence")
    memory = _memory_atom(atom_id="atom-memory")
    store.add_atom(evidence)
    store.add_atom(memory)

    with pytest.raises(sqlite3.IntegrityError):
        store.add_edge(
            _edge(
                edge_id="edge-evidence-source",
                source_atom_id=evidence.atom_id,
                target_atom_id=memory.atom_id,
            )
        )

    with pytest.raises(sqlite3.IntegrityError):
        store.add_edge(
            _edge(
                edge_id="edge-evidence-target",
                source_atom_id=memory.atom_id,
                target_atom_id=evidence.atom_id,
            )
        )


def test_store_axiom_transaction_rolls_back_invalid_field_batch(store: SQLiteMemoryStore) -> None:
    atom = _memory_atom(atom_id="atom-batch")
    bad_edge = _edge(
        edge_id="edge-batch-invalid",
        source_atom_id=atom.atom_id,
        target_atom_id="missing-target",
    )

    with pytest.raises(sqlite3.IntegrityError):
        with store.transaction():
            store.add_atom(atom)
            store.add_edge(bad_edge)

    assert store.list_atoms("subject-a") == ()
    assert store.list_edges("subject-a") == ()


def test_store_axiom_activation_cache_replace_is_atomic(store: SQLiteMemoryStore) -> None:
    atom = _memory_atom(atom_id="atom-cache")
    store.add_atom(atom)
    store.replace_activation_cache("subject-a", {atom.atom_id: 0.4}, updated_at=1.0)

    with pytest.raises(sqlite3.IntegrityError):
        store.replace_activation_cache("subject-a", {"missing-atom": 0.9}, updated_at=2.0)

    assert store.list_activation_cache("subject-a") == {atom.atom_id: 0.4}

    with pytest.raises(sqlite3.IntegrityError):
        store.replace_activation_cache("subject-a", {atom.atom_id: 1.5}, updated_at=3.0)

    assert store.list_activation_cache("subject-a") == {atom.atom_id: 0.4}


def test_store_axiom_rejects_malformed_persisted_content_payload(store: SQLiteMemoryStore) -> None:
    store.conn.execute(
        """
        INSERT INTO memory_atoms (
            atom_id, subject_id, atom_kind, content, confidence, salience, source_atom_ids, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("atom-bad-content", "subject-a", "memory", '"not-an-object"', 0.9, 0.8, "[]", 1.0),
    )
    store.conn.commit()

    with pytest.raises(ValueError, match="expected JSON object"):
        store.get_atom("atom-bad-content")


def test_store_axiom_rejects_invalid_persisted_evidence_payload(store: SQLiteMemoryStore) -> None:
    store.conn.execute(
        """
        INSERT INTO memory_atoms (
            atom_id, subject_id, atom_kind, content, confidence, salience, source_atom_ids, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "atom-bad-evidence",
            "subject-a",
            "evidence",
            '{"event_kind":"unknown","role":"user","text":"x","payload":{}}',
            1.0,
            0.2,
            "[]",
            1.0,
        ),
    )
    store.conn.commit()

    with pytest.raises(ValueError, match="invalid event_kind"):
        store.get_atom("atom-bad-evidence")


def test_store_axiom_rejects_invalid_persisted_episode_time_span(store: SQLiteMemoryStore) -> None:
    store.conn.execute(
        """
        INSERT INTO memory_atoms (
            atom_id, subject_id, atom_kind, content, confidence, salience, source_atom_ids, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "atom-bad-episode",
            "subject-a",
            "episode",
            '{"text":"episode","time_span":{"start":5.0,"end":4.0},"referents":[]}',
            0.9,
            0.8,
            "[]",
            1.0,
        ),
    )
    store.conn.commit()

    with pytest.raises(ValueError, match="time_span end must be >="):
        store.get_atom("atom-bad-episode")


def test_store_axiom_rejects_non_list_source_atom_ids(store: SQLiteMemoryStore) -> None:
    store.conn.execute(
        """
        INSERT INTO memory_atoms (
            atom_id, subject_id, atom_kind, content, confidence, salience, source_atom_ids, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "atom-bad-sources",
            "subject-a",
            "memory",
            '{"text":"memory","referents":[]}',
            0.9,
            0.8,
            '{"not":"a-list"}',
            1.0,
        ),
    )
    store.conn.commit()

    with pytest.raises(ValueError, match="expected JSON array"):
        store.get_atom("atom-bad-sources")
