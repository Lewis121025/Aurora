from __future__ import annotations

import sqlite3


def apply_migrations(connection: sqlite3.Connection) -> None:
    with connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS turns("
            "turn_id TEXT PRIMARY KEY, "
            "session_id TEXT NOT NULL, "
            "speaker TEXT NOT NULL, "
            "text TEXT NOT NULL, "
            "created_at REAL NOT NULL"
            ")"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS fragments("
            "fragment_id TEXT PRIMARY KEY, "
            "turn_id TEXT NOT NULL, "
            "text TEXT NOT NULL, "
            "created_at REAL NOT NULL, "
            "salience REAL NOT NULL"
            ")"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS traces("
            "trace_id TEXT PRIMARY KEY, "
            "turn_id TEXT NOT NULL, "
            "mode TEXT NOT NULL, "
            "intensity REAL NOT NULL, "
            "created_at REAL NOT NULL"
            ")"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS associations("
            "association_id TEXT PRIMARY KEY, "
            "source_fragment_id TEXT NOT NULL, "
            "target_fragment_id TEXT NOT NULL, "
            "weight REAL NOT NULL, "
            "created_at REAL NOT NULL"
            ")"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS relation_moments("
            "moment_id TEXT PRIMARY KEY, "
            "session_id TEXT NOT NULL, "
            "turn_id TEXT NOT NULL, "
            "tone TEXT NOT NULL, "
            "summary TEXT NOT NULL, "
            "created_at REAL NOT NULL"
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
            "CREATE TABLE IF NOT EXISTS snapshots("
            "snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "phase TEXT NOT NULL, "
            "self_view REAL NOT NULL, "
            "world_view REAL NOT NULL, "
            "openness REAL NOT NULL, "
            "updated_at REAL NOT NULL"
            ")"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS runtime_meta("
            "meta_key TEXT PRIMARY KEY, "
            "meta_value TEXT NOT NULL"
            ")"
        )
    _ensure_column(
        connection,
        table_name="fragments",
        column_name="narrative_weight",
        definition="REAL NOT NULL DEFAULT 0.0",
    )


def _ensure_column(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    definition: str,
) -> None:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    names = {str(row[1]) for row in rows}
    if column_name in names:
        return
    with connection:
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")
