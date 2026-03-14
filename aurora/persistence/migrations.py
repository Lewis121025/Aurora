from __future__ import annotations

import sqlite3


def apply_migrations(connection: sqlite3.Connection) -> None:
    with connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS turns("
            "turn_id TEXT PRIMARY KEY, "
            "relation_id TEXT NOT NULL, "
            "session_id TEXT NOT NULL, "
            "speaker TEXT NOT NULL, "
            "text TEXT NOT NULL, "
            "created_at REAL NOT NULL, "
            "reply_to_turn_id TEXT"
            ")"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS fragments("
            "fragment_id TEXT PRIMARY KEY, "
            "turn_id TEXT NOT NULL, "
            "relation_id TEXT NOT NULL, "
            "surface TEXT NOT NULL, "
            "touch_channels TEXT NOT NULL, "
            "salience REAL NOT NULL, "
            "vividness REAL NOT NULL, "
            "unresolvedness REAL NOT NULL, "
            "activation REAL NOT NULL, "
            "created_at REAL NOT NULL, "
            "last_touched_at REAL NOT NULL"
            ")"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS traces("
            "trace_id TEXT PRIMARY KEY, "
            "fragment_id TEXT NOT NULL, "
            "relation_id TEXT NOT NULL, "
            "channel TEXT NOT NULL, "
            "intensity REAL NOT NULL, "
            "persistence REAL NOT NULL, "
            "created_at REAL NOT NULL, "
            "last_touched_at REAL NOT NULL"
            ")"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS associations("
            "edge_id TEXT PRIMARY KEY, "
            "src_fragment_id TEXT NOT NULL, "
            "dst_fragment_id TEXT NOT NULL, "
            "kind TEXT NOT NULL, "
            "weight REAL NOT NULL, "
            "evidence_count INTEGER NOT NULL, "
            "created_at REAL NOT NULL, "
            "last_touched_at REAL NOT NULL"
            ")"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS threads("
            "thread_id TEXT PRIMARY KEY, "
            "relation_id TEXT NOT NULL, "
            "fragment_ids TEXT NOT NULL, "
            "motif_channels TEXT NOT NULL, "
            "coherence REAL NOT NULL, "
            "tension REAL NOT NULL, "
            "synopsis TEXT NOT NULL, "
            "created_at REAL NOT NULL, "
            "last_rewoven_at REAL NOT NULL"
            ")"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS knots("
            "knot_id TEXT PRIMARY KEY, "
            "relation_id TEXT NOT NULL, "
            "fragment_ids TEXT NOT NULL, "
            "channel TEXT NOT NULL, "
            "density REAL NOT NULL, "
            "heat REAL NOT NULL, "
            "created_at REAL NOT NULL, "
            "last_touched_at REAL NOT NULL"
            ")"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS relation_moments("
            "moment_id TEXT PRIMARY KEY, "
            "relation_id TEXT NOT NULL, "
            "session_id TEXT NOT NULL, "
            "user_turn_id TEXT NOT NULL, "
            "aurora_turn_id TEXT NOT NULL, "
            "user_channels TEXT NOT NULL, "
            "user_move TEXT NOT NULL, "
            "aurora_move TEXT NOT NULL, "
            "boundary_signal REAL NOT NULL, "
            "resonance_score REAL NOT NULL, "
            "note TEXT NOT NULL, "
            "created_at REAL NOT NULL"
            ")"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS relation_formations("
            "relation_id TEXT PRIMARY KEY, "
            "trust REAL NOT NULL, "
            "familiarity REAL NOT NULL, "
            "reciprocity REAL NOT NULL, "
            "boundary_tension REAL NOT NULL, "
            "repairability REAL NOT NULL, "
            "active_thread_ids TEXT NOT NULL, "
            "active_knot_ids TEXT NOT NULL, "
            "last_contact_at REAL NOT NULL"
            ")"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS orientations("
            "relation_id TEXT PRIMARY KEY, "
            "self_orientation REAL NOT NULL, "
            "world_orientation REAL NOT NULL, "
            "relation_orientation REAL NOT NULL, "
            "narrative_tilt REAL NOT NULL, "
            "updated_at REAL NOT NULL"
            ")"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS phase_transitions("
            "transition_id TEXT PRIMARY KEY, "
            "from_phase TEXT NOT NULL, "
            "to_phase TEXT NOT NULL, "
            "reason TEXT NOT NULL, "
            "created_at REAL NOT NULL"
            ")"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS metabolic_state("
            "state_id INTEGER PRIMARY KEY CHECK(state_id = 1), "
            "phase TEXT NOT NULL, "
            "sleep_need REAL NOT NULL, "
            "current_relation_id TEXT, "
            "active_thread_ids TEXT NOT NULL, "
            "active_knot_ids TEXT NOT NULL, "
            "last_transition_at REAL NOT NULL"
            ")"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS runtime_meta("
            "meta_key TEXT PRIMARY KEY, "
            "meta_value TEXT NOT NULL"
            ")"
        )
