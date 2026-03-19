from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from aurora.memory.experts import GLOBAL_EXPERT_ID, ExpertRecord
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


def _expert(
    *,
    expert_id: str,
    expert_kind: str,
    subject_id: str = "subject-a",
) -> ExpertRecord:
    return ExpertRecord(
        expert_id=expert_id,
        subject_id=subject_id,
        expert_kind=cast(Any, expert_kind),
        centroid=(),
        atom_count=0,
        updated_at=1.0,
        last_activated_at=0.0,
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


def test_store_axiom_expert_activation_cache_replace_is_atomic(store: SQLiteMemoryStore) -> None:
    atom = _memory_atom(atom_id="atom-cache")
    store.add_atom(atom)
    store.add_expert(_expert(expert_id=GLOBAL_EXPERT_ID, expert_kind="global"))
    store.add_expert(_expert(expert_id="expert-topic", expert_kind="topic"))
    store.set_atom_memberships("subject-a", atom.atom_id, primary_expert_id="expert-topic", secondary_expert_ids=(GLOBAL_EXPERT_ID,))
    store.replace_expert_activation_cache("subject-a", GLOBAL_EXPERT_ID, {atom.atom_id: 0.4}, updated_at=1.0)

    with pytest.raises(sqlite3.IntegrityError):
        store.replace_expert_activation_cache("subject-a", GLOBAL_EXPERT_ID, {"missing-atom": 0.9}, updated_at=2.0)

    assert store.list_expert_activation_cache("subject-a", GLOBAL_EXPERT_ID) == {atom.atom_id: 0.4}
    assert store.list_activation_cache("subject-a") == {atom.atom_id: 0.4}

    with pytest.raises(sqlite3.IntegrityError):
        store.replace_expert_activation_cache("subject-a", GLOBAL_EXPERT_ID, {atom.atom_id: 1.5}, updated_at=3.0)

    assert store.list_expert_activation_cache("subject-a", GLOBAL_EXPERT_ID) == {atom.atom_id: 0.4}


def test_store_axiom_rejects_non_global_secondary_membership(store: SQLiteMemoryStore) -> None:
    atom = _memory_atom(atom_id="atom-secondary")
    store.add_atom(atom)
    store.add_expert(_expert(expert_id=GLOBAL_EXPERT_ID, expert_kind="global"))
    store.add_expert(_expert(expert_id="expert-topic", expert_kind="topic"))

    with pytest.raises(sqlite3.IntegrityError, match="secondary memberships require a global expert"):
        store.conn.execute(
            """
            INSERT INTO expert_atoms (subject_id, expert_id, atom_id, membership_role)
            VALUES (?, ?, ?, 'secondary')
            """,
            ("subject-a", "expert-topic", atom.atom_id),
        )


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


def test_store_session_segments_round_trip_transcript_and_token_count(store: SQLiteMemoryStore) -> None:
    subject_id = "subject-session"
    session_id = "session-a"
    store.ensure_session(subject_id, session_id, started_at=1.0)
    segment = store.create_segment(subject_id, session_id, started_at=1.0)

    user = MemoryAtom(
        atom_id="atom-session-user",
        subject_id=subject_id,
        atom_kind="evidence",
        content=EvidenceContent(
            event_kind="user_turn",
            role="user",
            text="I live in Hangzhou.",
            payload={"session_id": session_id, "segment_id": segment.segment_id},
        ),
        confidence=1.0,
        salience=0.2,
        created_at=1.0,
    )
    assistant = MemoryAtom(
        atom_id="atom-session-assistant",
        subject_id=subject_id,
        atom_kind="evidence",
        content=EvidenceContent(
            event_kind="assistant_turn",
            role="assistant",
            text="ack",
            payload={"session_id": session_id, "segment_id": segment.segment_id},
        ),
        confidence=1.0,
        salience=0.2,
        created_at=1.000001,
    )
    store.add_atom(user)
    store.add_atom(assistant)
    store.append_session_event(subject_id, session_id, segment.segment_id, user.atom_id, token_count_increment=4)
    store.append_session_event(subject_id, session_id, segment.segment_id, assistant.atom_id, token_count_increment=1)

    transcript = store.list_segment_transcript(subject_id, session_id, segment.segment_id)
    open_segment = store.get_open_segment(subject_id, session_id)

    assert [item.role for item in transcript] == ["user", "assistant"]
    assert [item.text for item in transcript] == ["I live in Hangzhou.", "ack"]
    assert open_segment is not None
    assert open_segment.token_count == 5

    store.close_segment(subject_id, session_id, segment.segment_id, closed_at=2.0, closed_reason="finalize")
    store.close_session(subject_id, session_id, finalized_at=2.0)
    session = store.get_session(subject_id, session_id)
    assert session is not None
    assert session.finalized_at == 2.0


def _create_legacy_v4_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        INSERT INTO metadata(key, value) VALUES ('schema_version', 'field-v4');

        CREATE TABLE memory_atoms (
            atom_id TEXT PRIMARY KEY CHECK(atom_id <> ''),
            subject_id TEXT NOT NULL CHECK(subject_id <> ''),
            atom_kind TEXT NOT NULL CHECK(atom_kind IN ('evidence', 'memory', 'episode', 'inhibition')),
            content TEXT NOT NULL,
            confidence REAL NOT NULL CHECK(confidence BETWEEN 0.0 AND 1.0),
            salience REAL NOT NULL CHECK(salience BETWEEN 0.0 AND 1.0),
            source_atom_ids TEXT NOT NULL,
            created_at REAL NOT NULL,
            CHECK(atom_kind <> 'evidence' OR source_atom_ids = '[]'),
            UNIQUE(subject_id, atom_id)
        );

        CREATE TABLE memory_edges (
            edge_id TEXT PRIMARY KEY CHECK(edge_id <> ''),
            subject_id TEXT NOT NULL CHECK(subject_id <> ''),
            source_atom_id TEXT NOT NULL,
            target_atom_id TEXT NOT NULL,
            influence REAL NOT NULL CHECK(influence BETWEEN -1.0 AND 1.0),
            confidence REAL NOT NULL CHECK(confidence BETWEEN 0.0 AND 1.0),
            created_at REAL NOT NULL,
            CHECK(source_atom_id <> target_atom_id),
            FOREIGN KEY(subject_id, source_atom_id) REFERENCES memory_atoms(subject_id, atom_id),
            FOREIGN KEY(subject_id, target_atom_id) REFERENCES memory_atoms(subject_id, atom_id)
        );

        CREATE TABLE activation_cache (
            subject_id TEXT NOT NULL CHECK(subject_id <> ''),
            atom_id TEXT NOT NULL,
            activation REAL NOT NULL CHECK(activation BETWEEN 0.0 AND 1.0),
            updated_at REAL NOT NULL,
            PRIMARY KEY(subject_id, atom_id),
            FOREIGN KEY(subject_id, atom_id) REFERENCES memory_atoms(subject_id, atom_id)
        );
        """
    )
    conn.execute(
        """
        INSERT INTO memory_atoms (
            atom_id, subject_id, atom_kind, content, confidence, salience, source_atom_ids, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("atom-legacy", "subject-a", "memory", '{"text":"legacy memory","referents":[]}', 0.9, 0.8, "[]", 1.0),
    )
    conn.commit()
    conn.close()


def test_store_schema_upgrade_bootstraps_experts_without_losing_facts(tmp_path: Path) -> None:
    db_path = tmp_path / "aurora.db"
    _create_legacy_v4_db(db_path)

    store = SQLiteMemoryStore(str(db_path))
    assert store.conn.execute("SELECT value FROM metadata WHERE key = 'schema_version'").fetchone()[0] == "field-v6"
    assert store.get_atom("atom-legacy") is not None
    experts = store.list_experts("subject-a")
    assert len(experts) == 2
    assert {expert.expert_kind for expert in experts} == {"global", "topic"}
    assert store.primary_expert_id("subject-a", "atom-legacy") is not None
    assert store.list_expert_atom_ids("subject-a", GLOBAL_EXPERT_ID, membership_role="secondary") == ("atom-legacy",)
    assert store.expert_token_overlap("subject-a", ("legacy",))
    store.close()


def test_store_schema_upgrade_bootstraps_evidence_only_subject_with_global_only(tmp_path: Path) -> None:
    db_path = tmp_path / "aurora.db"
    _create_legacy_v4_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM memory_atoms")
    conn.execute(
        """
        INSERT INTO memory_atoms (
            atom_id, subject_id, atom_kind, content, confidence, salience, source_atom_ids, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "atom-evidence-only",
            "subject-a",
            "evidence",
            '{"event_kind":"user_turn","role":"user","text":"observed trace","payload":{}}',
            1.0,
            0.2,
            "[]",
            1.0,
        ),
    )
    conn.commit()
    conn.close()

    store = SQLiteMemoryStore(str(db_path))
    experts = store.list_experts("subject-a")
    assert len(experts) == 1
    assert experts[0].expert_kind == "global"
    assert store.primary_expert_id("subject-a", "atom-evidence-only") is None
    assert store.list_expert_atom_ids("subject-a", GLOBAL_EXPERT_ID, membership_role="secondary") == ()
    store.close()


def test_store_schema_upgrade_rolls_back_if_bootstrap_fails(tmp_path: Path) -> None:
    db_path = tmp_path / "aurora.db"
    _create_legacy_v4_db(db_path)

    class FailingBootstrapStore(SQLiteMemoryStore):
        def _bootstrap_expert_state(self) -> None:
            super()._bootstrap_expert_state()
            raise RuntimeError("bootstrap failed")

    with pytest.raises(RuntimeError, match="bootstrap failed"):
        FailingBootstrapStore(str(db_path))

    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT value FROM metadata WHERE key = 'schema_version'").fetchone()[0] == "field-v4"
    assert conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'experts'").fetchone() is None
    conn.close()


def test_store_rejects_incomplete_v6_expert_state_on_startup(tmp_path: Path) -> None:
    db_path = tmp_path / "aurora.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        INSERT INTO metadata(key, value) VALUES ('schema_version', 'field-v6');

        CREATE TABLE memory_atoms (
            atom_id TEXT PRIMARY KEY CHECK(atom_id <> ''),
            subject_id TEXT NOT NULL CHECK(subject_id <> ''),
            atom_kind TEXT NOT NULL CHECK(atom_kind IN ('evidence', 'memory', 'episode', 'inhibition')),
            content TEXT NOT NULL,
            confidence REAL NOT NULL CHECK(confidence BETWEEN 0.0 AND 1.0),
            salience REAL NOT NULL CHECK(salience BETWEEN 0.0 AND 1.0),
            source_atom_ids TEXT NOT NULL,
            created_at REAL NOT NULL,
            CHECK(atom_kind <> 'evidence' OR source_atom_ids = '[]'),
            UNIQUE(subject_id, atom_id)
        );
        """
    )
    conn.execute(
        """
        INSERT INTO memory_atoms (
            atom_id, subject_id, atom_kind, content, confidence, salience, source_atom_ids, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("atom-half", "subject-a", "memory", '{"text":"half migrated","referents":[]}', 0.9, 0.8, "[]", 1.0),
    )
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="incomplete expert state"):
        SQLiteMemoryStore(str(db_path))


def test_store_accepts_v6_evidence_only_subject_with_global_only(tmp_path: Path) -> None:
    db_path = tmp_path / "aurora.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        INSERT INTO metadata(key, value) VALUES ('schema_version', 'field-v6');

        CREATE TABLE memory_atoms (
            atom_id TEXT PRIMARY KEY CHECK(atom_id <> ''),
            subject_id TEXT NOT NULL CHECK(subject_id <> ''),
            atom_kind TEXT NOT NULL CHECK(atom_kind IN ('evidence', 'memory', 'episode', 'inhibition')),
            content TEXT NOT NULL,
            confidence REAL NOT NULL CHECK(confidence BETWEEN 0.0 AND 1.0),
            salience REAL NOT NULL CHECK(salience BETWEEN 0.0 AND 1.0),
            source_atom_ids TEXT NOT NULL,
            created_at REAL NOT NULL,
            CHECK(atom_kind <> 'evidence' OR source_atom_ids = '[]'),
            UNIQUE(subject_id, atom_id)
        );

        CREATE TABLE experts (
            subject_id TEXT NOT NULL CHECK(subject_id <> ''),
            expert_id TEXT NOT NULL CHECK(expert_id <> ''),
            expert_kind TEXT NOT NULL CHECK(expert_kind IN ('global', 'topic')),
            centroid TEXT NOT NULL,
            atom_count INTEGER NOT NULL CHECK(atom_count >= 0),
            updated_at REAL NOT NULL,
            last_activated_at REAL NOT NULL,
            PRIMARY KEY(subject_id, expert_id)
        );
        """
    )
    conn.execute(
        """
        INSERT INTO memory_atoms (
            atom_id, subject_id, atom_kind, content, confidence, salience, source_atom_ids, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "atom-evidence-only",
            "subject-a",
            "evidence",
            '{"event_kind":"user_turn","role":"user","text":"observed trace","payload":{}}',
            1.0,
            0.2,
            "[]",
            1.0,
        ),
    )
    conn.execute(
        """
        INSERT INTO experts (
            subject_id, expert_id, expert_kind, centroid, atom_count, updated_at, last_activated_at
        ) VALUES (?, ?, 'global', '[]', 0, 1.0, 0.0)
        """,
        ("subject-a", GLOBAL_EXPERT_ID),
    )
    conn.commit()
    conn.close()

    store = SQLiteMemoryStore(str(db_path))
    assert tuple(expert.expert_kind for expert in store.list_experts("subject-a")) == ("global",)
    store.close()
